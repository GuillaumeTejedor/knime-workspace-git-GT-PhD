import numpy as np

def pareto_front(workflows_metrics):

    """
    Apply pareto front to detect which workflows_metrics are selected by pareto front.

    :param workflows_metrics: Dataframe where each row represent a workflow and each column a metric.
    :return: Boolean mask (True: this workflow is selected by pareto front; False: this workflow is not selected by pareto front)
    """

    workflows_metrics = np.asarray(workflows_metrics)
    n = workflows_metrics.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if is_pareto[i]:
            dominates = np.all(workflows_metrics <= workflows_metrics[i], axis=1) & np.any(workflows_metrics < workflows_metrics[i], axis=1)
            is_pareto[i] = not np.any(dominates)
    return is_pareto