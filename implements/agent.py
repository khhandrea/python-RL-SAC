class Agent:
    def load(self):
        raise NotImplementedError('Should be implemented to load the model')
    
    def save(self):
        raise NotImplementedError('Should be implemented to save the model')
    
    def select_action(self):
        raise NotImplementedError('Should be implemented to select the action')
    
    def update_parameters(self):
        raise NotImplementedError('Should be implemented to update the parameters')