from typing import Callable, List, Dict
from argparse import Namespace

from ax.service.ax_client import AxClient

def optimize_hparams(name: str, trainer: Callable, args: Namespace, hparams_config: List[Dict], n_trials: int = 25) -> tuple:
    def evaluate_hparams(trainer: Callable, args: Namespace, hparams: dict):
        # parse hparams into args
        for key, value in hparams.items():
            setattr(args, key, value)
        
        print("Evaluating hparams: \n {}".format(hparams))
        best_val_acc, _ = trainer(args)

        return {"acc": (best_val_acc, 0.0)}

    from ax.service.ax_client import AxClient
    from ax.utils.notebook.plotting import render, init_notebook_plotting
    # init_notebook_plotting()
    ax_client = AxClient()

    ax_client.create_experiment(
        name=name,
        parameters=hparams_config,
        objective_name="acc",
        minimize=False
    )

    # start optimizing
    for _ in range(n_trials):
        hparams, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index, raw_data=evaluate_hparams(trainer, args, hparams))

    best_hparams, values = ax_client.get_best_parameters()
    return best_hparams, values