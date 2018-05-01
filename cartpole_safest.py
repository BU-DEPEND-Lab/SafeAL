import gym
import re
import numpy as np
import sys
from grids import grids
import MDP

from mdp import mdp
from scipy import sparse
import os

env = gym.make('CartPole-v0')

policy = None

file = open('./mdp.adv', 'r')
for line in file.readlines():
    if re.search('(?<=Strategy:)', line) is not None:
        policy = np.zeros([402, 200])
        continue
    elif policy is None:
        continue
    else:
        line_ = line.split(' ')
        s = int(line_[0])
        t = 0
        for i in range(1, len(line_)):
            if i%2 == 1:
                t = 200 - int(line_[i])
                continue
            elif i%2 == 0:
                a = int(line_[i])
                for j in range(t, 200):
                    policy[s][j] = a


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
        self.set_unsafe()

    def set_unsafe(self):
        self.M.unsafe = []
        for s in self.M.S[:-2]:
            coords = self.index_to_coord(s) 
	    if (coords[0] <= 0 and coords[2] <=  1)  or ((coords[0]) >= (self.grids[0] - 1) and (coords[2] >= self.grids[2] - 2)):

	        self.M.unsafes.append(s)

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
		    #if sum(streak) < 80:

                        #paths = list()
			#using = 0
                        #self.M.starts = list()
                        pass
		    episodes = 0
		if using > maxepisodes/3 and i_episode > maxepisodes/2:
		    break

	    	print "Episode {} finished after {} timecoords {} win:{} use:{}".format(i_episode, t+1, tag, win, using)

	
        file = open('./data/start', 'w')
	for s in self.M.S:
            if starts[s]:
                print("start from %d" % s)
	        file.write(str(s) + '\n')
                self.M.starts.append(s)
        file.close()

    
        for a in range(len(self.M.A)):
            for s in range(len(self.M.S)):
                tot = np.sum(self.M.T[a][s])
                if tot == 0.0:
                    self.M.T[a][s, s] = 0.0
            self.M.T[a] = sparse.bsr_matrix(self.M.T[a]).astype(np.float16)        
            self.M.T[a] = sparse.diags(1.0/self.M.T[a].sum(axis = 1).A.ravel()).dot(self.M.T[a]).todense()

        self.M.set_initial_transitions()

        for s in self.M.unsafes:
            for a in self.M.A:
                self.M.T[a][s] = 0.0 * self.M.T[a][s]
                self.M.T[a][s, -1] = 1.0
    
	file = open('./data/mdp', 'w')
        for s in self.M.S:
            for a in self.M.A:
                for s_ in self.M.S:
                    file.write(str(s) + ' ' 
                                + str(a) + ' ' 
                                + str(s_) + ' ' 
                                + str(self.M.T[a][s, s_]) + '\n')
        file.close()

        file = open('./data/state_space', 'w')
        file.write('states\n' + str(len(self.M.S)) + '\nactions\n' + str(len(self.M.A)))
        file.close()
            
        os.system(str(os.path.dirname(os.path.realpath(__file__))) + '/prism-4.4.beta-src/src/demos/mdp ' + str(os.path.dirname(os.path.realpath(__file__))))

    def test(self):
        o = env.reset()
        env.render()
        t = raw_input('cut!!!!')
        for t in range(self.steps):
            s = int(self.observation_to_index(o))
            a = int(policy[s][t])
            o1, rew, done, info = env.step(a)
            env.render()
            o = o1
            

if __name__== '__main__':
    cartpole = cartpole()
    #cartpole.run_tool_box()
    cartpole.test()
