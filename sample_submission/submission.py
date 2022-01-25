# from Steven/Goktug's repo import AbstractOptimizer #TODO
from dac4automlcomp.run_experiments import run_experiment


class RandomOptimizer():
    '''
    #TODO
    '''

    def __init__(self,):
        """

        Parameters
        ----------
        """
        # AbstractOptimizer.__init__(self, api_config)
        ...

    def act(self,):
        """

        Parameters
        ----------

        Returns
        -------
        """
        return [1]

    def reset(self, task_info):
        """

        Parameters
        ----------
        """
        # Random search so don't do anything
        pass

    def save(self,):
        ...

    def load(self,):
        return self

if __name__ == "__main__":
    obj = RandomOptimizer()
    run_experiment(obj) 
