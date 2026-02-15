class MuZeroConfig:
    def __init__(self):
        # Environment
        self.env_name = 'PongNoFrameskip-v4'
        self.action_space_size = 6 # Pong has 6 actions
        self.input_shape = (4, 84, 84) # 4 stacked frames, 84x84
        
        # Self-Play
        self.num_actors = 1
        self.num_simulations = 50
        self.discount = 0.997
        
        # Root exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        
        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        
        # Training
        self.training_steps = int(1000e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = 128
        self.num_unroll_steps = 5
        self.td_steps = 10
        
        # Optimization
        self.lr_init = 0.01
        self.weight_decay = 1e-4
        self.momentum = 0.9
        
        # Device
        self.device = 'cpu'
        
        # Training loop
        self.num_episodes = 1000
