# from Steven/Goktug's repo import AbstractPolicy #TODO
from dac4automlcomp.run_experiments import run_experiment


class RandomPolicy():
    '''
    
    #TODO Flesh out the code below
    '''

    def __init__(self,):
        """

        Parameters
        ----------
        """
        # AbstractPolicy.__init__(self, api_config)
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
    obj = RandomPolicy()
    run_experiment(obj) 
