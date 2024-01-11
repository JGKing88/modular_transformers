from shapely.geometry import Polygon, Point
import random
import torch
import math
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
torch.manual_seed(1)
import multiprocessing

class PathDataset(torch.utils.data.Dataset):
    def __init__(self, velocity_sets, path_sets):
        self.velocity_sets = velocity_sets
        self.path_sets = path_sets

    def __len__(self):
        return len(self.velocity_sets)

    def __getitem__(self, idx):
      return {
          "inputs": torch.tensor(self.velocity_sets[idx]).float(),
          "labels": torch.tensor(self.path_sets[idx]).float()}

#Step 3: Intialize RNN with biological features
class BioRNN(nn.Module):
  def __init__(self, config):
    super(BioRNN, self).__init__()
    self.input_dim = config["input_dim"]
    self.hidden_dim = config["hidden_dim"]
    self.output_dim = config["output_dim"]
    self.tau_multiplier = config["tau_multiplier"]

    self.W_in = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
    self.W_rec = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
    self.W_out = nn.Linear(self.hidden_dim, self.output_dim, bias=False)
    self.b = nn.Parameter(torch.zeros((1, self.hidden_dim)))

  def __call__(self, input, noise):
    batch_size, time_steps, _ = input.shape
    x = torch.zeros((batch_size, self.hidden_dim)).to(input.device)
    outputs = torch.zeros((batch_size, time_steps, self.output_dim)).to(input.device)
    hidden_states = torch.zeros((batch_size, time_steps, self.hidden_dim)).to(input.device)

    for t in range(time_steps):
      inp = self.W_in(input[:,t,:])
      h = torch.tanh(x)
      rec = self.W_rec(h)
      x = x + self.tau_multiplier * (-x + inp + rec + self.b + noise[:,t,:])

      outputs[:,t,:] = self.W_out(x)
      hidden_states[:,t,:] = x
    
    l2_weights = [self.W_in.weight, self.W_out.weight]
    l2_weights = torch.cat([w.reshape(-1) for w in l2_weights])

    return outputs, hidden_states, l2_weights

  def initialize_weights(self):
    for name, param in self.named_parameters():
      if 'W_rec' in name:
        if 'weight' in name:
          nn.init.normal_(param, mean=config["rec_init"][0], std = config["rec_init"][1])
        if 'bias' in name:
          nn.init.zeros_(param)
      elif 'W_in' in name:
        if 'weight' in name:
          nn.init.normal_(param, mean=config["in_init"][0], std = config["in_init"][1])
        if 'bias' in name:
          nn.init.zeros_(param)
      elif 'W_out' in name:
        if 'weight' in name:
          nn.init.normal_(param, mean=config["out_init"][0], std = config["out_init"][1])
        if 'bias' in name:
          nn.init.zeros_(param)

class RegRNN(nn.Module):
  def __init__(self, config):
    super(RegRNN, self).__init__()
    input_dim = config["input_dim"]
    hidden_dim = config["hidden_dim"]
    output_dim = config["output_dim"]
    self.tau_multiplier = config["tau_multiplier"]

    self.W_in = nn.Linear(input_dim, hidden_dim, bias=False)
    self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.W_out = nn.Linear(hidden_dim, output_dim, bias=False)
    self.b = nn.Parameter(torch.zeros((1, hidden_dim)))

  def __call__(self, input, h_prev):
    inp = self.W_in(input)
    rec = self.W_rec(h_prev)
    h_current = h_prev + self.tau_multiplier * (-h_prev + inp + rec + self.b)
    h_current = torch.tanh(h_current)
    y = self.W_out(h_current)
    return y, h_current, self.W_in.weight, self.W_out.weight

  def initialize_weights(self):
    for name, param in self.named_parameters():
      if 'W_rec' in name:
        if 'weight' in name:
          nn.init.normal_(param, mean=config["rec_init"][0], std = config["rec_init"][1])
        if 'bias' in name:
          nn.init.zeros_(param)
      elif 'W_in' in name:
        if 'weight' in name:
          nn.init.normal_(param, mean=config["in_init"][0], std = config["in_init"][1])
        if 'bias' in name:
          nn.init.zeros_(param)
      elif 'W_out' in name:
        if 'weight' in name:
          nn.init.normal_(param, mean=config["out_init"][0], std = config["out_init"][1])
        if 'bias' in name:
          nn.init.zeros_(param)

def evaluate(model, loss_fn, dataloader):
  model.eval()
  total_loss = 0
  with torch.no_grad():
    for i, batch in enumerate(tqdm(dataloader)):
      inputs = batch["inputs"].to(device)
      labels = batch["labels"].to(device)
      bs, ts, _ = inputs.shape
      noises = torch.zeros(bs, ts, 1).to(device)
      outputs, hidden_state, l2_weights = model(inputs, noises)

      loss = loss_fn(outputs, labels)
      total_loss += loss.item()
  
  #return average loss per sample
  return total_loss / len(dataloader)
  
  

def reg_evaluate(model, loss_fn, dataloader):
  model.eval()
  total_loss = 0
  with torch.no_grad():
    for i, batch in enumerate(tqdm(dataloader)):
      num_steps = batch["inputs"].shape[1]
      batch_size = batch["inputs"].shape[0]

      inputs = batch["inputs"].reshape(num_steps, batch_size, -1).to(device)
      labels = batch["labels"].reshape(num_steps, batch_size, -1).to(device)

      hidden_state = torch.zeros((batch_size, config["hidden_dim"])).to(device)

      for input, label in zip(inputs, labels):
        output, hidden_state, W_in, W_out = model(input, hidden_state)
        loss = loss_fn(output, label)
        total_loss += loss.item()
  
  #return average loss per sample
  return total_loss / ( num_steps * len(dataloader))

def scaled_frobenius_norm(matrix):
    squared_elements = matrix ** 2
    sum_of_squares = torch.sum(squared_elements)
    num_elements = matrix.numel()
    return sum_of_squares / num_elements
  
def bio_train(model, trainloader, validloader, config, path, name):

  optimizer = Adam(model.parameters(), lr=config["lr"])
  loss_fn = MSELoss()

  model.train()

  l2_multiplier = 1
  metabolic_reg_multiplier = 1

  for epoch in range(config["epochs"]):
    epoch_loss = 0
    for i, batch in enumerate(tqdm(trainloader)):
      optimizer.zero_grad()

      labels = batch["labels"].to(device)
      inputs = batch["inputs"].to(device)
      
      noises = torch.normal(0.0, data_config["noise_variance"], size=(config["batch_size"], config["time_steps"], config["hidden_dim"])).to(device)
      outputs, hidden_states, l2_weights = model(inputs, noises)

      loss = loss_fn(outputs, labels)
      epoch_loss += loss.item()
      wandb.log({"train_loss": loss.item()})
                  
      l2_reg = scaled_frobenius_norm(l2_weights)
      l2_multiplier = loss.item() / l2_reg * ((len(trainloader) - i) / len(trainloader))
      metabolic_reg = scaled_frobenius_norm(hidden_states)
      metabolic_reg_multiplier = (loss.item() / metabolic_reg) * (i / len(trainloader))
      if metabolic_reg_multiplier > loss / 3: 
        metabolic_reg_multiplier = loss / 3
      loss += l2_reg * l2_multiplier + metabolic_reg * metabolic_reg_multiplier
      loss.backward()
      optimizer.step()

      if i % 50 == 0:
        eval_loss = evaluate(model, loss_fn, validloader)
        print("Epoch: {} Step: {} Train Loss: {} Valid Loss: {}".format(epoch+1, i, epoch_loss / (i+1), eval_loss))
        wandb.log({"eval_loss": eval_loss})
        model.train()
      
    wandb.log({"epoch": epoch+1, "avg_train_loss": epoch_loss / len(trainloader)})
    torch.save(model.state_dict(), f"{path}{name}.pt")


def reg_train(model, trainloader, validloader, config):

    optimizer = Adam(model.parameters(), lr=config["lr"])
    loss_fn = MSELoss()

    train_losses = []
    valid_losses = []

    model.train()

    for epoch in range(config["epochs"]):
        epoch_loss = 0
        for i, batch in enumerate(tqdm(trainloader)):
            optimizer.zero_grad()
            num_steps = batch["inputs"].shape[1]
            batch_size = batch["inputs"].shape[0]

            inputs = batch["inputs"].reshape(num_steps, batch_size, -1).to(device)
            labels = batch["labels"].reshape(num_steps, batch_size, -1).to(device)

            hidden_state = torch.zeros((batch_size, config["hidden_dim"])).to(device)

            batch_loss = 0
            no_reg_loss = 0
            for input, label in zip(inputs, labels):
                output, hidden_state, W_in, W_out = model(input, hidden_state)
                loss = loss_fn(output, label)
                batch_loss += loss
                no_reg_loss += loss.item()
            
            epoch_loss += no_reg_loss

            train_loss = no_reg_loss / num_steps
            train_losses.append(train_loss)

            wandb.log({"train_loss": train_loss})

            batch_loss.backward()
            optimizer.step()

            if i % 50 == 0:
                eval_loss = reg_evaluate(model, loss_fn, validloader)
                wandb.log({"eval_loss": eval_loss})
                valid_losses.append(eval_loss)
                model.train()
                print("Epoch: {} Train Loss: {} Eval Loss: {}".format(epoch+1, train_loss, eval_loss))
    
        wandb.log({"epoch": epoch, "avg_train_loss": epoch_loss / len(trainloader)})   
   

class RNNWithLinearPerTimestep(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNWithLinearPerTimestep, self).__init__()
        # RNN Layer
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Linear output layer
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # Pass the input through the RNN layer
        # out shape: [batch_size, seq_length, hidden_size]
        hidden_states, _ = self.rnn(input)

        # Reshape output for linear layer
        # New shape: [batch_size * seq_length, hidden_size]
        batch_size, seq_length, hidden_size = hidden_states.shape
        out = hidden_states.reshape(batch_size * seq_length, hidden_size)

        # Pass each time step through the linear layer
        # out shape: [batch_size * seq_length, output_size]
        out = self.linear(out)

        # Reshape back to out: [batch_size, seq_length, output_size]
        out = out.reshape(batch_size, seq_length, -1)
        # hidden_states: [batch_size, seq_length, hidden_size]
        return out, hidden_states

def evaluate_builtin(model, loss_fn, dataloader):
  model.eval()
  total_loss = 0
  with torch.no_grad():
    for i, batch in enumerate(tqdm(dataloader)):
      num_steps = batch["inputs"].shape[1]
      batch_size = batch["inputs"].shape[0]

      inputs = batch["inputs"].reshape(batch_size, num_steps, -1).to(device)
      labels = batch["labels"].reshape(batch_size, num_steps, -1).to(device)

      outputs, _ = model(inputs)
      loss = loss_fn(outputs, labels)
      total_loss += loss.item()
  
  #return average loss per sample
  return total_loss / len(dataloader)

def train_builtin(model, trainloader, validloader, config, path, name):

    optimizer = Adam(model.parameters(), lr=config["lr"])
    loss_fn = MSELoss()

    train_losses = []
    valid_losses = []

    model.train()

    for epoch in range(config["epochs"]):
        epoch_loss = 0
        for i, batch in enumerate(tqdm(trainloader)):
            optimizer.zero_grad()
            num_steps = batch["inputs"].shape[1]
            batch_size = batch["inputs"].shape[0]

            inputs = batch["inputs"].reshape(batch_size, num_steps, -1).to(device)
            labels = batch["labels"].reshape(batch_size, num_steps, -1).to(device)

            outputs, _ = model(inputs)
            loss = loss_fn(outputs, labels)
            
            epoch_loss += loss.item()

            train_loss = loss.item()
            train_losses.append(train_loss)

            wandb.log({"train_loss": train_loss})

            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                eval_loss = evaluate_builtin(model, loss_fn, validloader)
                wandb.log({"eval_loss": eval_loss})
                valid_losses.append(eval_loss)
                model.train()
                print("Epoch: {} Train Loss: {} Eval Loss: {}".format(epoch+1, train_loss, eval_loss))
    
        wandb.log({"epoch": epoch+1, "avg_train_loss": epoch_loss / len(trainloader)})  
        torch.save(model.state_dict(), f"{path}{name}.pt") 
    


class BioLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BioLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gate layers
        self.Wii = nn.Linear(input_size, hidden_size)
        self.Whi = nn.Linear(hidden_size, hidden_size)
        self.bi = nn.Parameter(torch.zeros(hidden_size))

        # Forget gate layers
        self.Wif = nn.Linear(input_size, hidden_size)
        self.Whf = nn.Linear(hidden_size, hidden_size)
        self.bf = nn.Parameter(torch.zeros(hidden_size))

        # Cell gate layers
        self.Wig = nn.Linear(input_size, hidden_size)
        self.Whg = nn.Linear(hidden_size, hidden_size)
        self.bg = nn.Parameter(torch.zeros(hidden_size))

        # Output gate layers
        self.Wio = nn.Linear(input_size, hidden_size)
        self.Who = nn.Linear(hidden_size, hidden_size)
        self.bo = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, init_states):
        """
        forward method for the LSTM cell.
        """
        h_t, c_t = init_states

        i_t = torch.sigmoid(self.Wii(x) + self.Whi(h_t) + self.bi)
        f_t = torch.sigmoid(self.Wif(x) + self.Whf(h_t) + self.bf)
        g_t = torch.tanh(self.Wig(x) + self.Whg(h_t) + self.bg)
        o_t = torch.sigmoid(self.Wio(x) + self.Who(h_t) + self.bo)

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t
    
    def initialize_weights(self):
      for name, param in self.named_parameters():
        if 'W' in name:
          if 'h' in name:
            nn.init.normal_(param, mean=config["rec_init"][0], std = config["rec_init"][1])
          elif 'i' in name:
            nn.init.normal_(param, mean=config["in_init"][0], std = config["in_init"][1])
          if 'bias' in name:
            nn.init.zeros_(param)

class BioLSTM(nn.Module):
    def __init__(self, config):
        super(BioLSTM, self).__init__()
        self.cell = BioLSTMCell(config["input_dim"], config["hidden_dim"])
        self.hidden_size = config["hidden_dim"]
        self.output_size = config["output_dim"]

        self.W_out = nn.Linear(self.hidden_size, config["output_dim"])
        self.b_c = nn.Parameter(torch.zeros((1, self.hidden_size)))
        self.b_h = nn.Parameter(torch.zeros((1, self.hidden_size)))
        self.tau_multiplier = config["tau_multiplier"]

    def forward(self, x, noises):
        """
        forward method for the LSTM.
        """
        bs, seq_sz, _ = x.size()

        h_t = torch.zeros(bs, self.hidden_size).to(x.device)
        c_t = torch.zeros(bs, self.hidden_size).to(x.device)

        hidden_states = torch.zeros(bs, seq_sz, self.hidden_size).to(x.device)
        outputs = torch.zeros(bs, seq_sz, self.output_size).to(x.device)

        for t in range(seq_sz):
            x_t = x[:, t, :]
            new_h_t, c_t = self.cell(x_t, (h_t, c_t))
            h_t = h_t + self.tau_multiplier * (-h_t + new_h_t + noises[:,t,:] + self.b_h)
            hidden_states[:,t,:] = h_t
            outputs[:,t,:] = self.W_out(h_t)
        
        l2_weights = [self.cell.Wii.weight, self.cell.Wif.weight, self.cell.Wig.weight, self.cell.Wio.weight]
        l2_weights = torch.cat([w.reshape(-1) for w in l2_weights])

        return outputs, hidden_states, l2_weights

    def initialize_weights(self):
      for name, param in self.named_parameters():
        if 'W' in name:
          if 'h' in name:
            nn.init.normal_(param, mean=config["rec_init"][0], std = config["rec_init"][1])
          elif 'i' in name:
            nn.init.normal_(param, mean=config["in_init"][0], std = config["in_init"][1])
          elif 'out' in name:
            nn.init.normal_(param, mean=config["out_init"][0], std = config["out_init"][1])
          if 'bias' in name:
            nn.init.zeros_(param)
      self.cell.initialize_weights()

class SemiBioLSTM(nn.Module):
    def __init__(self, config):
        super(SemiBioLSTM, self).__init__()
        self.cell = BioLSTMCell(config["input_dim"], config["hidden_dim"])
        self.hidden_size = config["hidden_dim"]
        self.output_size = config["output_dim"]

        self.W_out = nn.Linear(self.hidden_size, config["output_dim"])
        self.b_c = nn.Parameter(torch.zeros((1, self.hidden_size)))
        self.b_h = nn.Parameter(torch.zeros((1, self.hidden_size)))
        self.tau_multiplier = config["tau_multiplier"]

    def forward(self, x, noises):
        """
        forward method for the LSTM.
        """
        bs, seq_sz, _ = x.size()

        h_t = torch.zeros(bs, self.hidden_size).to(x.device)
        c_t = torch.zeros(bs, self.hidden_size).to(x.device)

        hidden_states = torch.zeros(bs, seq_sz, self.hidden_size).to(x.device)
        outputs = torch.zeros(bs, seq_sz, self.output_size).to(x.device)

        for t in range(seq_sz):
            x_t = x[:, t, :]
            h_t, c_t = self.cell(x_t, (h_t, c_t))
            hidden_states[:,t,:] = h_t
            outputs[:,t,:] = self.W_out(h_t)
        
        l2_weights = [self.cell.Wii.weight, self.cell.Wif.weight, self.cell.Wig.weight, self.cell.Wio.weight]
        l2_weights = torch.cat([w.reshape(-1) for w in l2_weights])

        return outputs, hidden_states, l2_weights
    
    def initialize_weights(self):
      for name, param in self.named_parameters():
        if 'W' in name:
          if 'h' in name:
            nn.init.normal_(param, mean=config["rec_init"][0], std = config["rec_init"][1])
          elif 'i' in name:
            nn.init.normal_(param, mean=config["in_init"][0], std = config["in_init"][1])
          elif 'out' in name:
            nn.init.normal_(param, mean=config["out_init"][0], std = config["out_init"][1])
          if 'bias' in name:
            nn.init.zeros_(param)
      self.cell.initialize_weights()

# Function to check if a point is inside the polygon
def is_inside_polygon(point, polygon):
  return polygon.contains(point)

# Function to resample direction based on Brownian motion
def resample_speed(current_speed,step_size):
  return abs(current_speed + np.random.normal(scale=np.sqrt(step_size/5)))

def brownian_walk_2d(num_steps, step_size, diffusion_coefficient, border_vertices, zero_prob = 0.9):
  border_polygon = Polygon(border_vertices)

  # Initialize arrays to store particle positions, speeds, and directions
  x = np.zeros(num_steps)
  y = np.zeros(num_steps)
  speeds = np.zeros(num_steps)
  directions = np.zeros(num_steps)
  delta_directions = np.zeros(num_steps)
  # Initial direction and position
  prev_direction = np.random.uniform(0, 2 * np.pi)
  speed = 0
  #x[0], y[0] = np.random.uniform(0, 5), np.random.uniform(0, 5)
  # Simulate Brownian motion
  for i in range(1, num_steps):
      # Resample direction based on Brownian motion
      theta = np.random.normal(scale=np.sqrt(2* step_size))
      direction = (prev_direction + theta) % (2*np.pi)
      speed = resample_speed(speed, step_size)
      # Calculate dx and dy components from the direction
      dx = speed * np.cos(direction)
      dy = speed * np.sin(direction)

      if np.random.rand() < zero_prob:  # 90% probability for speed to be 0
        dx, dy, speed = 0, 0, 0

      # Update particle position
      new_x = x[i - 1] + dx
      new_y = y[i - 1] + dy

      # Check if the new position is inside the border, adjust if necessary
      new_point = Point(new_x, new_y)

      count = 0
      while not is_inside_polygon(new_point, border_polygon):
          # If the new position is outside the border, resample the direction

          theta = np.random.normal(scale=np.sqrt(2* step_size)) / 10 * count
          direction = prev_direction + theta
            
          count += 1

          direction = direction % (2*np.pi)
          speed = resample_speed(speed, step_size)
          # Calculate dx and dy components from the direction
          dx = speed * np.cos(direction)
          dy = speed * np.sin(direction)
          # Update particle position
          new_x = x[i - 1] + dx
          new_y = y[i - 1] + dy

          # Check if the new position is inside the border, adjust if necessary
          new_point = Point(new_x, new_y)

      x[i] = x[i - 1] + dx
      y[i] = y[i - 1] + dy

      # Calculate speed and store direction
      speed = np.sqrt(dx**2 + dy**2)
      speeds[i] = speed
      directions[i] = direction / (2*np.pi)
      delta_directions[i] = theta / (2*np.pi)
 
      # Update previous direction
      prev_direction = direction

  return x,y,speeds,directions, delta_directions


def generate_dataset(num_steps, step_size, diffusion_coefficient, border_vertices, dataset_len):
  #initialize
  velocity_sets = np.zeros((dataset_len, num_steps, 2))
  path_sets = np.zeros((dataset_len, num_steps, 2))

  for i in tqdm(range(dataset_len)):
    x, y, speeds, directions, delta_directions = brownian_walk_2d(num_steps,step_size,diffusion_coefficient,border_vertices)

    path = np.stack([x, y], axis=1)
    velocities = np.stack([speeds, directions], axis=1)

    velocity_sets[i] = velocities
    path_sets[i] = path

  return [velocity_sets, path_sets, dataset_len]

def make_data_sets(data_config, path):
  num_steps = data_config["time_steps"]  # Number of steps
  step_size = data_config["step_size"]  # Step size
  diffusion_coefficient = data_config["diffusion_coefficient"] # Diffusion coefficient
  dataset_len = data_config["dataset_len"]
  # Define the border polygon (you can change the vertices to define your border)
  border_vertices = data_config["vertices"]

  # with multiprocessing.Pool() as pool:
  results = generate_dataset(num_steps, step_size, diffusion_coefficient, border_vertices, dataset_len)
  np.save(f"{path}trainset_{dataset_len}_velocities_absolute_partial7", results[0])
  np.save(f"{path}trainset_{dataset_len}_paths_absolute_partial7", results[1])

  dataset_len = data_config["validset_len"]
  results = generate_dataset(num_steps, step_size, diffusion_coefficient, border_vertices, dataset_len)
  np.save(f"{path}validset_{dataset_len}_velocities_absolute_partial7", results[0])
  np.save(f"{path}validset_{dataset_len}_paths_absolute_partial7", results[1])

   

if __name__ == "__main__":

    wandb.login(key="a338f755915cccd861b14f29bf68601d8e1ec2c9")

    # Config
    data_config = {
            "time_steps": 500,
            "step_size": 0.1,
            "diffusion_coefficient": 0.1,
            "noise_variance": 0.05,
            "dataset_len": 230000,
            "validset_len": 30000,
            "batch_size": 1000,
            "granularity": 0.001,
            "vertices": [(-1, -1), (1, -1), (1, 1), (-1, 1)]}

    hidden_dim = 100
    input_dim = 2

    config = {"input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": 2,
            "epochs": 10000,
            "tau_multiplier": 1/10,
            "time_steps": 500,
            "rec_init": (0.0, 1.5**2 / hidden_dim),
            "out_init": (0.0, 0.0),
            "in_init": (0.0, 1/input_dim),
            "lr": 0.001}
    
    path = '/om2/user/jackking/modular_transformers/modular_transformers/dynamics/'

    dataset_len = data_config["dataset_len"]
    velocity_sets = np.load(f"{path}trainset_{dataset_len}_velocities_absolute_final.npy")
    path_sets = np.load(f"{path}trainset_{dataset_len}_paths_absolute_final.npy")
    dataset = PathDataset(velocity_sets, path_sets)
    trainloader = DataLoader(dataset, batch_size=data_config["batch_size"], shuffle=True)

    dataset_len = data_config["validset_len"]
    velocity_sets = np.load(f"{path}validset_{dataset_len}_velocities_absolute_final.npy")
    path_sets = np.load(f"{path}validset_{dataset_len}_paths_absolute_final.npy")
    dataset = PathDataset(velocity_sets, path_sets)
    validloader = DataLoader(dataset, batch_size=data_config["batch_size"], shuffle=False)

    config = {**config, **data_config}
    load_name = "bio_rnn_2kbatch_230kdata_100hidden_old"
    name = "bio_rnn_2kbatch_230kdata_100hidden"

    run = wandb.init(
        # Set the project where this run will be logged
        project="gridcells",
        # Track hyperparameters and run metadata
        config={**config},
        name=name,
    )

    # model = RegRNN(config)
    # model.initialize_weights()

    # model = RNNWithLinearPerTimestep(input_dim, hidden_dim, 2, 1)
# 
    model = BioRNN(config)

    # model = SemiBioLSTM(config)
    # model = BioLSTM(config)

    # model = BioLSTM(config)
    model.load_state_dict(torch.load(f"{path}{load_name}.pt"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # model.initialize_weights()

    bio_train(model, trainloader, validloader, config, path, name)
    # train_builtin(model, trainloader, validloader, config, path, name)
# 
    torch.save(model.state_dict(), f"{path}{name}.pt")





