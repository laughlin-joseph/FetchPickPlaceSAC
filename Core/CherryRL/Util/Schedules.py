from abc import ABC

class Schedule(ABC):
    def __init__(self, total_steps, final_val, initial_val=1.0):
        self.total_steps = total_steps
        self.final_val = final_val
        self.initial_val = initial_val

class LinearSchedule(Schedule):
    def __init__(self, total_steps, final_val, initial_val=1.0):
        super().__init__(total_steps, final_val, initial_val)

    def get_step_val(self, t):
        decay_factor = min(float(t) / self.total_steps, 1.0)
        
        return self.initial_val + decay_factor * (self.final_val - self.initial_val)
    
class ExponentialSchedule(Schedule):
    def __init__(self, total_steps, final_val, initial_val=1.0):
        super().__init__(total_steps, final_val, initial_val)
    
    def get_step_val(self, t):
        decay_factor = min((self.final_val / self.initial_val) ** (t / self.total_steps),
                                (self.final_val / self.initial_val))
        
        return self.initial_val * decay_factor