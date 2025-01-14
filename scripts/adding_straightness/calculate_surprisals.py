import torch

from transformers import GPT2Tokenizer

from minicons import scorer

import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM

from torchviz import make_dot

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

path = "/om2/user/jackking/modular_transformers/scripts/adding_straightness/data"


def run(num_perturbations, perturb_location):
    """
    Calculate surprisal values using Phi-2 model:
    - Loads perturbed sentences
    - Computes token-level surprisal
    - Handles different perturbation types
    - Saves results to disk
    """
    # path_to_dict = f"/om2/user/jackking/modular_transformers/scripts/adding_straightness/perturb_straight_byact_results_{perturb_location}.pkl"
    path_to_dict = f"/om2/user/jackking/modular_transformers/scripts/adding_straightness/data/perturb_{num_perturbations}x_straight_results_{perturb_location}.pkl"
    results = pickle.load(open(path_to_dict, "rb"))

    save_path = (
        f"{path}/phi-2_surprisal_perturb_{num_perturbations}x_{perturb_location}.pkl"
    )
    # save_path = f"{path}/phi-2_surprisal_perturb_{perturb_location}_byact.pkl"

    new_dict = {}

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", trust_remote_code=True
    )
    model = scorer.IncrementalLMScorer(model, tokenizer=tokenizer, device=device)

    sentences = results["normal"]["sentences"]
    surprisals_tensor = torch.zeros(len(sentences), len(sentences[1]))

    for i, sentence in enumerate(sentences):
        surprisals = model.token_score(sentence, surprisal=True)[0]
        for j, token in enumerate(surprisals):
            surprisals_tensor[i][j] = token[1]

    new_dict["normal"] = surprisals_tensor

    # types = ["parallel", "orthogonal", "random", "straightened", "curved", "prev_parallel", "neg_prev_parallel", "on_path", "off_path"]
    # multipliers = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]

    types = ["straighter", "curved"]
    multipliers = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3]

    for multiplier in multipliers:

        new_dict[multiplier] = {}
        for perturb_type in types:
            sentences = results[multiplier][perturb_type]["sentences"]
            surprisals_tensor = torch.zeros(len(sentences), len(sentences[1]))
            for i, sentence in enumerate(sentences):
                # surprisals = torch.tensor(model.sequence_score(sentence, reduction = lambda x: -x.sum(0)))
                surprisals = model.token_score(sentence, surprisal=True)[0]

                for j, token in enumerate(surprisals):
                    surprisals_tensor[i][j] = token[1]

            new_dict[multiplier][perturb_type] = surprisals_tensor

        pickle.dump(new_dict, open(save_path, "wb"))
        print(f"Saved {multiplier}")


if __name__ == "__main__":
    num_perturbations = 1
    perturb_location = "blocks.5.hook_resid_post"
    run(num_perturbations, perturb_location)

    perturb_location = "blocks.15.hook_resid_post"
    run(num_perturbations, perturb_location)

    perturb_location = "blocks.30.hook_resid_post"
    run(num_perturbations, perturb_location)

    perturb_location = "blocks.20.hook_resid_post"
    run(num_perturbations, perturb_location)

    num_perturbations = 3
    perturb_location = "blocks.5.hook_resid_post"
    run(num_perturbations, perturb_location)

    perturb_location = "blocks.15.hook_resid_post"
    run(num_perturbations, perturb_location)

    perturb_location = "blocks.30.hook_resid_post"
    run(num_perturbations, perturb_location)

    perturb_location = "blocks.20.hook_resid_post"
    run(num_perturbations, perturb_location)
