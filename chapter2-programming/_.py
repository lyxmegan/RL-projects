import numpy as np
import sys

def nonstationary_env(num_bandits, init_q_estimate, init_q_star, std_q_star, std_change_q_star, 
                      eps, alpha, steps, runs):

    average_reward_sample_average = np.zeros(steps)
    average_reward_const_step_size = np.zeros(steps)
    ration_of_optimal_sample_average = np.zeros(steps)
    ration_of_optimal_const_step_size = np.zeros(steps)

    for run in range(runs):
        q_estimate_sample_ave = np.ones(num_bandits) * init_q_estimate
        q_star = np.ones(num_bandits) * init_q_star

        average_reward_per_step_sample_ave = []
        ration_optimal_per_step_sample_ave = []
        count_action_sample_ave = np.zeros(num_bandits)

        total_reward_sample_ave = 0.0
        optimal_action_count_sample_ave = 0
        
        avg_reward_per_step_const_step_size = []
        ration_optimal_per_step_const = []
        q_estimate_const = np.ones(num_bandits) * init_q_estimate

        total_reward_const = 0.0
        optimal_action_count_const = 0
        
        for step in range(steps):
            #greedy
            if np.random.uniform() > eps:
                sample_ave_act = np.argmax(q_estimate_sample_ave)
                const_step_size_act = np.argmax(q_estimate_const)
            else:
                sample_ave_act = np.random.randint(0, num_bandits)
                const_step_size_act = sample_ave_act
                
            #get the optimal action: 
            optimal_action = np.argmax(q_star)
            #check if the action picked equal to optimal action
            if sample_ave_act == optimal_action:
                optimal_action_count_sample_ave += 1
            if const_step_size_act == optimal_action:
                optimal_action_count_const += 1

            ration_optimal_per_step_sample_ave.append(int(sample_ave_act == optimal_action))
            ration_optimal_per_step_const.append(int(const_step_size_act == optimal_action))
            sample_ave_reward = np.random.normal(q_star[sample_ave_act], std_q_star)
            const_step_size_reward = np.random.normal(q_star[const_step_size_act], std_q_star)
            
            total_reward_sample_ave += sample_ave_reward
            total_reward_const += const_step_size_reward

            average_reward_per_step_sample_ave.append(sample_ave_reward)
            avg_reward_per_step_const_step_size.append(const_step_size_reward)
            
            count_action_sample_ave[sample_ave_act] += 1
            
            q_estimate_sample_ave[sample_ave_act] += (sample_ave_reward - q_estimate_sample_ave[sample_ave_act])
            q_estimate_const[const_step_size_act] += alpha * (const_step_size_reward - q_estimate_const[const_step_size_act])
            
            q_star += np.random.randn(num_bandits) * std_change_q_star

        average_reward_sample_average += np.array(average_reward_per_step_sample_ave)
        average_reward_const_step_size += np.array(ration_optimal_per_step_sample_ave)
        ration_of_optimal_sample_average += np.array(avg_reward_per_step_const_step_size)
        ration_of_optimal_const_step_size += np.array(ration_optimal_per_step_const)

    one = average_reward_sample_average / runs
    two = average_reward_const_step_size / runs
    three = ration_of_optimal_sample_average / runs
    four = ration_of_optimal_const_step_size / runs
    np.savetxt(sys.argv[1], (one, two, three, four))


if __name__ == '__main__':
    num_bandits = 10
    init_q_estimate = 0.
    init_q_star = 0.
    std_q_star = 1.
    std_change_q_star = 0.01
    eps = 0.1
    alpha = 0.1
    steps = 10000
    runs = 300
    nonstationary_env(num_bandits, init_q_estimate, init_q_star, std_q_star, std_change_q_star, eps,
                         alpha, steps, runs)
