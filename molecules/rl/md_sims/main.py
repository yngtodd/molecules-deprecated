import os
from reinforce import reinforce

# Define Parameters
num_trials = 1 # Runs 5 different RL batches
main_dir = "./RL_testing/" #Main directory where all files are stored
episodes = 3
sim_steps = 20000
traj_out_freq = 100
#sim_num = 5

# Create main directory
if not os.path.exists(main_dir):
    os.mkdir(main_dir)

for i in range(num_trials):
    trial_dir = main_dir + "trial_%i_results/" % i
    if not os.path.exists(trial_dir):
        os.mkdir(trial_dir)
    rl = reinforce(sim_steps=sim_steps,
                   traj_out_freq=traj_out_freq,
                   episodes=episodes,
                   output_dir=trial_dir)
    rl.main()
    print("Trials completed:", i + 1, "/", num_trials)
