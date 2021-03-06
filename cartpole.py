import sys
import MDP
from mdp import mdp
from grids import grids
import numpy as np
import scipy.optimize as optimize
from cvxopt import matrix, solvers
from scipy import sparse
import os
import time
import mdptoolbox
import math
import pylab
from apirl import apirl
from cegal import cegal

import ast
import gym
import re
import cProfile
import math
from grids import grids

# In order to run this script you must download and compile
# the following library: https://github.com/Svalorzen/AI-Toolbox
# Building it will create an MDP.so file which you can then
# include here.

env = gym.make('CartPole-v0')

class cartpole(grids, object):
    def __init__(self):
        if sys.version_info[0] >= 3:  
            super().__init__()
        else:
            super(cartpole, self).__init__()

        self.grids = [5, 4, 5, 4]

        threshes = [0.3, 0.4, 0.4, 0.4]
        self.threshes = list()
        for i in range(len(self.grids)):
            self.threshes.append([-1.0 * threshes[i]])
            for j in range(self.grids[i] - 2):            
                self.threshes[i].append(self.threshes[i][-1] +  2 * threshes[i]/(self.grids[i] - 2))
            assert len(self.threshes[i]) == self.grids[i] - 1 

        num_S = 1
        for i in self.grids:
            num_S *= i
        num_S += 2

        num_A = 2

        self.M = mdp(num_S, num_A = 2)

        self.set_unsafe()

        self.steps = 200
        self.maxepisodes = 100000

        self.opt = None

    def set_initial_opt(self):
        self.opt = {}
        
        self.opt['policy'] = self.read_policy_file(path = './data/init_cartpole')
        self.M.set_policy(self.opt['policy'])
        self.opt['mu'] = self.M.expected_features_manual()[-2]
        self.opt['theta'] = list()
    
    
    def set_unsafe(self):
        self.M.unsafe = []
        for s in self.M.S[:-2]:
            coords = self.index_to_coord(s) 
	    if (coords[0] <= 0 and coords[2] <=  1)  or ((coords[0]) >= (self.grids[0] - 1) and (coords[2] >= self.grids[2] - 2)):

	        self.M.unsafes.append(s)

    def check_transitions(self):
        for a in self.M.A:
            if isinstance(self.M.T[a], np.ndarray) is False:
                self.M.T[a] = np.array(self.M.T[a])
            for s in self.M.S:
                p_tot = self.M.T[a][s].sum() 
                if p_tot < 1.0:
                    self.M.T[a][s, s] = 1.0 - p_tot
                if p_tot > 1.0:
                    self.M.T[a][s] = self.M.T[a][s] * 1.0/p_tot



    def build_features(self):
        f = 30
        self.M.features = np.zeros([len(self.M.S), f])
        feature_states = []
        for i in range(f):
            s = int(len(self.M.S) * i/f)
            feature_states.append(s)
       
        for s in self.M.S: 
            coord = self.index_to_coord(s)
            y = coord[1] + coord[2] * self.grids[1] + coord[3] * self.grids[2] * self.grids[1]
            x = coord[0]
            for i in range(f):
                s_ = feature_states[i]
                coord_ = self.index_to_coord(s_)
                y_ = coord_[1] + coord_[2] * self.grids[1] + coord_[3] * self.grids[2] * self.grids[1]
                x_ = coord_[0]
                self.M.features[s, i] = math.exp(-0.25 * math.sqrt((1.0 * y - y_)**2 + (1.0 * x - x_)**2))
        


    def build_MDP_from_file(self):
        os.system('cp ./data/state_space_cartpole ./data/state_space')
        os.system('cp ./data/unsafe_cartpole ./data/unsafe')
        os.system('cp ./data/mdp_cartpole ./data/mdp')
        os.system('cp ./data/start_cartpole ./data/start')

        self.M.input()

        self.set_unsafe()

        self.check_transitions()

        self.build_features()
        
        self.M.set_initial_transitions()

        self.set_initial_opt()

        self.M.output()


    def learn_from_demo_file(self, steps = 200):
        learn = cegal(self.M, max_iter = 50, epsilon = 10)
        learn.exp_mu = learn.read_demo_file('./data/demo_cartpole') 
        print(learn.exp_mu)
        opt = super(cegal, learn).iteration(learn.exp_mu) 
        prob = learn.model_check(opt['policy'], steps)
        opt['prob'] = prob
    
        print("\n>>>>>>>>Apprenticeship Learning learnt policy weight vector:")
        print(opt['theta'])
        print("\nFeature vector margin: %f" % opt['diff'])
        print("\nGiven safety spec:\nP=? [U<= 200 ((position < -0.3 && angle < -20)||(position > 0.3 && angle > 20))]\n")
        print("\nPRISM model checking the probability of reaching unsafe states: %f\n" % prob)


        file = open('./data/log', 'a')
        file.write("\n>>>>>>>>Apprenticeship Learning learns a policy.\n")
        file.write("\nGiven safety spec:\nP=? [U<= 200 ((position < -0.3 && angle < -20)||(position > 0.3 && angle > 20))]\n")
        file.write("\nPRISM model checking the probability of reaching\
 the unsafe states: %f\n" % prob)
        file.close()

        while True:
            test = raw_input('\n1. Play learnt policy visually\n\
2. Run policy to collect statistical data\n3. Store policy\n4. Quit\nInput the selection:\n')
            if test == '1':
                self.episode(policy = opt['policy'], steps = steps)
            elif test == '2':
		file = open('./data/log', 'a')
		file.write('\nTest AL policy\n')
                file.close()
                self.test(policy = opt['policy'])
            elif test == '3':
                self.write_policy_file(policy = opt['policy'], path = './data/policy_cartpole')
            elif test == '4':
                break
            else:
                print("Invalid input")
        return opt

    
    def write_policy_file(self, policy = None, path = './data/policy_cartpole'):
        if policy is None:
            policy = self.M.policy

        os.system('rm ' + str(path))
        os.system('touch ' + str(path))
        file = open(str(path), 'w')
        for s in self.M.S:
            for a in self.M.A:
                file.write(str(policy[s, a]) + ' ')
            file.write('\n')
        file.close()
        
    def read_policy_file(self,  path = './data/policy_cartpole'):
        self.M.policy = np.zeros([len(self.M.S), len(self.M.A)])
        file = open(str(path), 'r')
        lines = file.readlines()
        for s in self.M.S:
            line = lines[s].split('\n')[0].split(' ')
            for a in self.M.A:
                self.M.policy[s, a] = float(line[a])
        file.close()
        self.M.set_policy()
        return self.M.policy
        
    def learn_from_policy_file(self):
        learn = apirl(self.M, max_iter = 50, epsilon = 10)
        self.read_policy_file()
        mus = self.M.expected_features_manual()
        learn.exp_mu = mus[-2]
        print(learn.exp_mu)
        opt = learn.iteration(learn.exp_mu) 
        return opt

    def episode(self, demo = False, safe = False, performance = False, policy = None, steps = None):
        if steps is None:
            steps = self.steps
        
        if policy is None:
            policy = self.M.policy

        unsafes = np.zeros([len(self.M.S)]).astype(bool)
        for s in self.M.unsafes:
            unsafes[s] = True

        count_down = 5
        
        cut = True
        if demo or safe or performance:
            cut = False

	o = env.reset()
	t = 0
	s = self.observation_to_index(o); 

        path = []   
	#print("state %d" % s)
	for t in xrange(steps):
            s =  self.observation_to_index(o); 
	    #print("state %d" % s)
     	    # Convert the observation into our own space
      	    # Select the best action according to the policy
     	    a = int(policy[s].argmax())
	    #print("action %d" % a)
                
       	    # Act
      	    o1, rew, done, info = env.step(a);
	    if done:
                if not safe or True:
                    print("End after steps %d" % t)
                    return path
                    
                    
      	    # See where we arrived
     	    s_ = self.observation_to_index(o1);

            path.append([t, s, a, s_])

            if unsafes[s_]:
                print("Reached unsafe state after steps %d" % t)
                if demo or safe:
                    return list()

            
     	    # Record information, and then run PrioritizedSweeping
            o = o1;

            if not(demo or performance or safe):
    	        env.render()
	    if cut:
	        while count_down > 0:
	            count_down -= 1
		    cut = raw_input("Click Enter!!!")
		    cut = False
		cut = raw_input("Click Enter one last time!!!")
        return path


    def demo(self, policy = None, episodes = 5000):
        os.system('rm ./data/demo_cartpole')
        os.system('toucn ./data/demo_cartpole')
        print("Generate %d episodes" % episodes)
        use = 0 
        file = open('./data/demo_cartpole', 'w')
        while use < 1000:
            path  = self.episode(demo = True, safe = True, policy = policy)
            episodes -= 1
            if len(path) >= self.steps - 2:
                use += 1
                print('%d validate demo have been generated' % use)
                print('Ratio: %f' % use/episodes)
                for t in range(len(path)):
                    file.write(str(path[t][0]) + ' ' 
                            + str(path[t][1]) + ' ' 
                            + str(path[t][2]) + ' '
                            + str(path[t][3]) + '\n')
        file.close()
        print("Use %d demos" % use)
        

    def test(self, policy = None, episodes = 1000):
        dead = 0.0 
        i_episode = 0

        while i_episode <= episodes:
            print("Episode %d" % i_episode)
            path = self.episode(safe = True, policy = policy)
            if len(path) == 0:
                dead += 1
            i_episode += 1
        dead = dead/episodes

        avg = 0
        i_episode = 0
        while i_episode < episodes:
            print("Episode %d" % i_episode)
            path  = self.episode(performance =True, policy = policy)
            avg += len(path)
            i_episode += 1
        avg /= episodes

        print('Unsafe ratio: %f' % dead)
        print("Average step length: %f" % avg)
        
        file = open('./data/log', 'a')
        file.write('Unsafe ratio: %f\n' % dead)
        file.write("Average step length: %f\n" % avg)
        file.close()

    def synthesize_from_demo_file(self, safety = 0.3, steps = 200, path = './data/demo_cartpole'):
        if safety is None:
            safety = self.safety
        if steps is None:
            steps = self.steps

        learn = cegal(self.M, max_iter = 50, safety = safety, steps = steps, epsilon = 10)
        exp_mu = learn.read_demo_file(path)

        opt, opt_ = self.synthesize(learn = learn, exp_mu = exp_mu, safety = safety, steps = steps)
        
        
        while True:
            n = raw_input('\n1. Test AL policy, \n2. Test Safety-Aware AL policy, \n3. Quit\n\
Input the seletion:\n')
            if n == '1':
		file = open('./data/log', 'a')
		file.write('\nTest AL policy\n')
                file.close()
                policy = opt_['policy'].copy()
            elif n == '2':
		file = open('./data/log', 'a')
		file.write('\nTest CEGAL policy\n')
                file.close()
                policy = opt['policy'].copy()
            elif n == '3':
                break
            else:
                print("Invalid")
                continue
            while True: 
                test = raw_input('\n1. Play learnt policy visually\n\
2. Run policy to collect statistical data\n3. Store policy\n4. Quit\nInput the selection:\n')
                if test == '1':
                    self.episode(policy = policy, steps = steps)
                elif test == '2':
                    self.test(policy = policy)
                elif test == '3':
                    if n == '1':
                        self.write_policy_file(policy = policy, path = './data/policy_cartpole')
                    else:
                        self.write_policy_file(policy = policy, path = './data/policy_cartpole_' + str(safety))
                elif test == '4':
                    break
                else:
                    print("Invalid input")

        return opt, opt_
            

    def synthesize(self, learn, exp_mu, opt = None, safety = None, steps = None):
        if safety is None:
            safety = self.safety
        if steps is None:
            steps = self.steps
        if opt is None:
            opt = self.opt

        opt, opt_ = learn.iteration(exp_mu = exp_mu, opt = opt, safety = safety, steps = steps)
        ## cegal.iteration returns SAAL and AL learning results
        ## opt = (diff, theta, policy, mu)

        print("\nLearning result for safety specification:\n")
        print("\nP<=" + str(safety) + " [U<= 200 ((position < -0.3 && angle < -20)||(position > 0.3 && angle > 20))]\n")

        print("\n>>>>>>>>Safety-Aware Apprenticeship Learning learnt policy weight vector:")
        print(opt['theta'])
        print("\nFeature vector margin: %f" % opt['diff'])
        print("\nPRISM model checking result: %f\n" % opt['prob'])
        
        file = open('./data/log', 'a')
        file.write("\n\n\nLearning result for safety specification:\n")
        file.write("\nP<=" + str(safety) + " [U<= 200 ((position < -0.3 && angle < -20)||(position > 0.3 && angle > 20))]\n")

        file.write("\n>>>>>>>>Safety-Aware Apprenticeship Learning")
        #print("\nFeature vector margin: %f" % opt['diff'])
        file.write("\nPRISM model checking result: %f\n" % opt['prob'])
        file.close()

        return opt, opt_

    def run_tool_box(self, steps = 200):
        paths = []

        self.M.starts = list()

        unsafes = np.zeros([len(self.M.S)]).astype(bool)
        for u in self.M.unsafes:
            unsafes[u] = True

        starts = np.zeros([len(self.M.S)]).astype(bool)
        
        self.M.T = list()
        for a in self.M.A:
            self.M.T.append(np.zeros([len(self.M.S), len(self.M.S)]))
        
        gamma = self.M.discount
        exp = MDP.SparseExperience(len(self.M.S), len(self.M.A));
        model = MDP.SparseRLModel(exp, gamma);
        solver = MDP.PrioritizedSweepingSparseRLModel(model, 0.1, 500);
        policy = MDP.QGreedyPolicy(solver.getQFunction());

        using = 0
	episodes=0
	win = 0
	streak = list()
	
        maxepisodes = 10000
        samples = 10000
	for i_episode in xrange(maxepisodes):
            if True:
                path = []
    		o = env.reset()
                env.render()

                s_i = self.observation_to_index(o)
                self.M.T[a][self.M.S[-2], s_i] += 1.0

		dead = False
                done = False

    		for t in xrange(steps):
        	    # Convert the observation into our own space
        	    s = self.observation_to_index(o);
        	    # Select the best action according to the policy
        	    a = policy.sampleAction(s)
        	    # Act
        	    o_, rew, done, info = env.step(a);
        	    # See where we arrived
        	    s_ = self.observation_to_index(o_);

                    self.M.T[a][s, s_] += 1.0
                    
                    path.append([t, s, a, s_])

                    if unsafes[s_]:
		        dead = True
                        
        	    if done:
                        if t < steps - 2:
                            dead = True
                        if i_episode > samples:
            		    break
                            
                    # Record information, and then run PrioritizedSweeping
        	    exp.record(s, a, s_, rew);
       		    model.sync(s, a, s_);
		    solver.stepUpdateQ(s, a);
		    solver.batchUpdateQ();

		    o = o_;

   		#   if render or i_episode == maxepisodes - 1:
      		#       env.render()

    		  
                # Here we have to set the reward since otherwise rewards are
                # always 1.0, so there would be no way for the agent to distinguish
                # between bad actions and good actions.
                tag = ' '
    		if done and i_episode > samples:
                    rew = -10
                    
                    if t < (steps - 2)/2:
                        rew = -100
                        tag ='xxx'
		        streak.append(0)
                    elif t >= steps - 2:
		        rew = 100
       		        tag = '###';
        	        streak.append(1)
      		        win += 1;
  		        if not dead:
                            using += 1
                            paths.append(path)
                            if not starts[s_i]:
                                #self.M.starts.append(s_i)
                                starts[s_i] = True
                                print("start from %d" % s)

                if done and i_episode < samples:
                    if dead:
                        rew = -100
                        tag ='xxx'
		        streak.append(0)
                    else:
		        rew = 100
       		        tag = '###';
        	        streak.append(1)
      		        win += 1;
                        if True:
                            using += 1
                            paths.append(path)
                            if not starts[s_i]:
                                starts[s_i] = True
                                print("start from %d" % s)


  		if len(streak) > 100:
      			streak.pop(0)

    		episodes +=1;
    		exp.record(s, a, s_, rew);
    		model.sync(s, a, s_);
    		solver.stepUpdateQ(s, a);
    		solver.batchUpdateQ();
                # If the learning process gets stuck in some local optima without
                # winning we just reset the learning. We don't want to try to change
                # what the agent has learned because this task is very easy to fail
                # when trying to learn something new (simple exploration will probably
                # just make the pole topple over). We just want to learn the correct
                # thing once and be done with it.
		if episodes == 100:
    		    #if sum(streak) < 90:
		    #	using -= episodes
		    #	demo_mu -= demo_mu_episodes 
		    #else:
		    #	for s_ in unsafe:
		    #		if s_[0] * coords[0] + s_[1] == s1:
		    #			using -= episodes
		    #			demo_mu -= demo_mu_episodes 
    		    if sum(streak) < 30:
	                exp = MDP.SparseExperience(len(self.M.S), len(self.M.A));
       			model = MDP.SparseRLModel(exp, gamma);
        		solver = MDP.PrioritizedSweepingSparseRLModel(model, 0.1, 500);
      	  		policy = MDP.QGreedyPolicy(solver.getQFunction());
		    if sum(streak) < 80:

                        #paths = list()
			#using = 0
                        #self.M.starts = list()
                        pass
		    episodes = 0
		if using > maxepisodes/3 and i_episode > maxepisodes/2:
		    break

	    	print "Episode {} finished after {} timecoords {} win:{} use:{}".format(i_episode, t+1, tag, win, using)

	
        file = open('./data/start_cartpole', 'w')
	for s in self.M.S:
            if starts[s]:
                print("start from %d" % s)
	        file.write(str(s) + '\n')
        file.close()

    
        for a in range(len(self.M.A)):
            for s in range(len(self.M.S)):
                tot = np.sum(self.M.T[a][s])
                if tot == 0.0:
                    self.M.T[a][s, s] = 0.0
            self.M.T[a] = sparse.bsr_matrix(self.M.T[a]).astype(np.float16)        
            self.M.T[a] = sparse.diags(1.0/self.M.T[a].sum(axis = 1).A.ravel()).dot(self.M.T[a]).todense()

	file = open('./data/mdp_cartpole', 'w')
        for s in self.M.S:
            for a in self.M.A:
                for s_ in self.M.S:
                    file.write(str(s) + ' ' 
                                + str(a) + ' ' 
                                + str(s_) + ' ' 
                                + str(self.M.T[a][s, s_]) + '\n')
        file.close()

        file = open('./data/unsafe_cartpole', 'w')
        for s in self.M.unsafes:
            file.write(str(s) + '\n')
        file.close()

        file = open('./data/state_space_cartpole', 'w')
        file.write('states\n' + str(len(self.M.S)) + '\nactions\n' + str(len(self.M.A)))
        file.close()

        file = open('./data/demo_cartpole', 'w')
        for path in paths:
            for t in range(len(path)):
                file.write(str(path[t][0]) + ' ' 
                            + str(path[t][1]) + ' ' 
                            + str(path[t][2]) + ' '
                            + str(path[t][3]) + '\n')
        file.close()


if __name__ == "__main__":
    cartpole = cartpole()
    cartpole.run_tool_box()
    cartpole.build_MDP_from_file()

    opt = cartpole.learn_from_demo_file()
        
    safety = opt['prob']
    safety = int(safety * 10) - 1
    safety = safety/10.0
    safety = 0.05
    
    cartpole.synthesize_from_demo_file(safety = safety, steps = 200)

