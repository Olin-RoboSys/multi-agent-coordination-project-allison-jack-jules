import sys
sys.path.insert(0, './scripts')
from utils import Position, dist
import networkx as nx
from PathPlanning import RoomMap, find_path

class GroundControlSystem():
    def __init__(self, agent_list=None, task_list=None, env=None):
        
        self._agent_list = agent_list
        self._task_list = task_list
        self._task_assignment = None
        self._agent_paths = dict()
        self._env = env
        
        grid_range_x = [self._env["dimensions"]["x_min"]*0.1, self._env["dimensions"]["x_max"]*0.1]
        grid_range_y = [self._env["dimensions"]["y_min"]*0.1, self._env["dimensions"]["y_max"]*0.1]
        print(grid_range_x, grid_range_y)
        num_nodes_x = int(self._env["dimensions"]["x_max"] - self._env["dimensions"]["x_min"] + 1)
        num_nodes_y = int(self._env["dimensions"]["y_max"] - self._env["dimensions"]["y_min"] + 1)
        self.room_map = RoomMap(grid_range_x, grid_range_y,
                            obst_nodes=self._env["obstacles"],
                            num_nodes_x=num_nodes_x, num_nodes_y=num_nodes_y)

    # getters and setters -------------------------------------------------

    def set_task_list(self, task_list):
        self._task_list = task_list


    def add_agents(self, agent_list):
        """add agents - already initialized with goal location, etc."""
        self._agent_list = agent_list

    
    def get_task_assignment(self, draw=False):
        """return the task assignment and draw if required"""
        
        j=0
        colors = ['b', 'g', 'r', 'm', 'y', 'c']

        for agent, task in self._task_assignment.items():
            if type(task) == list:
                for i in range(len(task)):
                    if i > 0:
                        self.G.add_edge(task[i-1].drop_id, task[i].pick_id, color=colors[j])
                        self.G.add_edge(task[i].pick_id, task[i].drop_id, color=colors[j])
                    else:
                        self.G.add_edge(agent._id, task[i].pick_id, color=colors[j])
                        self.G.add_edge(task[i].pick_id, task[i].drop_id, color=colors[j])
            else:
                self.G.add_edge(agent._id, task.pick_id, color=colors[j])
                self.G.add_edge(task.pick_id, task.drop_id, color=colors[j])
            j+=1
        
        if draw:
            color_scheme = nx.get_edge_attributes(self.G,'color').values()
            pos=nx.get_node_attributes(self.G,'pos')
            nx.draw(self.G,pos,with_labels = True, edge_color=color_scheme)

        return self._task_assignment
    

    def set_task_graph(self, draw=False):
        """creates a directed graph based on the agents and task list using networkx"""
        self.G=nx.DiGraph()

        # add agent start locations:
        for a in self._agent_list.values():
            self.G.add_node(a._id, pos=(a.get_pos().x, a.get_pos().y))

        # add pick and drop locations of tasks:
        for t in self._task_list.values():
            self.G.add_node(t.pick_id, pos=(t.pick_loc.x, t.pick_loc.y))
            self.G.add_node(t.drop_id, pos=(t.drop_loc.x, t.drop_loc.y))
            # add edges connecting pick and drop locations
            self.G.add_edge(t.pick_id, t.drop_id)

        pos=nx.get_node_attributes(self.G,'pos')

        if draw:
            nx.draw(self.G,pos,with_labels = True)


    def get_agent_paths(self):
        return self._agent_paths
    
    # ---------------------------------------------------------------------


    def create_task_assignment(self):
        """make task assignment"""

        # ##################################################################################
        # STEP 1: Create a cost matrix for cost from each agent to each pick location
        # and then from each agent to complete each task (i.e. pick to drop)
        # ##################################################################################

        cost_to_pick_loc = dict()
        cost_of_full_task = dict()

        # create the cost matrices:
        for a in self._agent_list.values():
            for t in self._task_list.values():
                # compute distance from agent a to pick location of task t
                c_to_p = dist(a.get_pos(), t.pick_loc)
                # print(f'{a._id} is [{a.get_pos().x}, {a.get_pos().y}]')

                # compute distance from pick location to drop location of task t
                c_to_d = dist(t.pick_loc, t.drop_loc)
                #  print(f'cost for {t.pick_id} -> {t.drop_id} == {c_to_d}')
                
                # arrange into dict
                cost_to_pick_loc[(a._id, t.pick_id)] = c_to_p
                # cost_of_full_task[(a._id, (t.pick_id, t.drop_id))] = c_to_p + c_to_d
                cost_of_full_task[(a._id, (t.pick_id, t.drop_id))] = c_to_p + c_to_d
        
        # print(cost_to_pick_loc)
        # print("--------------------------------------\n")

        # ##################################################################################
        # STEP 2: Assign based on minimum cost
        # ##################################################################################
        self._task_assignment = dict()
        num_agents = len(self._agent_list)

        copy_of_cost_to_pick_loc = cost_to_pick_loc.copy() # create a copy of the cost matrix

        while len(self._task_assignment) < num_agents: # assign to all agents
            ass_t = min(copy_of_cost_to_pick_loc, key=copy_of_cost_to_pick_loc.get)
            

            # check if agent OR if task's pick loc has been assigned
            agent = self._agent_list[ass_t[0]]
            task = self._task_list[ass_t[1]]


            if agent in self._task_assignment or task in self._task_assignment.values():
                del copy_of_cost_to_pick_loc[ass_t] # delete it, if so
            else:
                self._task_assignment[agent] = self._task_list[task.pick_id]
                del copy_of_cost_to_pick_loc[ass_t]

        # for agent, task in self._task_assignment.items():
        #     print(f'Agent {agent._id} : {task.pick_id} -> {task.drop_id}') % DEBUG

        # ##################################################################################
        # STEP 3: Assign based on minimum cost
        # ##################################################################################
        new_cost_to_pick_loc = dict()
        # create the cost matrices:
        for a in self._agent_list.values():
            for t in self._task_list.values():
                if t not in self._task_assignment.values():
                    # compute distance from agent a to pick location of task t
                    c_to_p = dist(a.get_pos(), t.pick_loc) 
                    # obtain cost of previous task
                    prev_t = self._task_assignment[a]
                    c_prev = cost_of_full_task[(a._id, (prev_t.pick_id, prev_t.drop_id))]
                    # add previous task cost to current task
                    c_to_p += c_prev
                    # compute distance from pick location to drop location of task t
                    c_to_d = dist(t.pick_loc, t.drop_loc)

                    # arrange into dict
                    new_cost_to_pick_loc[(a._id, t.pick_id)] = c_to_p

        # print(new_cost_to_pick_loc) % DEBUG


        # ##################################################################################
        # STEP 4: Assign again based on minimum cost
        # ##################################################################################
        num_tasks = len(self._task_list)
        pending_tasks = num_tasks - len(self._task_assignment)
        new_assigned_agents = []
        new_assigned_tasks = []

        copy_of_cost_to_pick_loc = new_cost_to_pick_loc.copy() # create a copy of the cost matrix
        i=0
        while i < pending_tasks: # make sure you cover all tasks
            if len(copy_of_cost_to_pick_loc) == 0:
                break
            else:
                ass_t = min(copy_of_cost_to_pick_loc, key=copy_of_cost_to_pick_loc.get)
            
            agent = self._agent_list[ass_t[0]]
            task = self._task_list[ass_t[1]]

            # check if agent OR if task's pick loc has been assigned
            if agent in new_assigned_agents or task in new_assigned_tasks:
                del copy_of_cost_to_pick_loc[ass_t] # delete it, if so
            else:
                # print(f'Minimum assignment is {ass_t} with cost: {copy_of_cost_to_pick_loc[ass_t]}') % DEBUG
                i += 1
                # convert task assignment value to a list to accommodate multiple tasks
                first_task = self._task_assignment[agent]
                self._task_assignment[agent] = [first_task]
                self._task_assignment[agent].append(self._task_list[task.pick_id])
                
                del copy_of_cost_to_pick_loc[ass_t]
                
                # update processed agents and tasks
                new_assigned_agents.append(agent)
                new_assigned_tasks.append(task)
                

        for agent, task in self._task_assignment.items():
            if type(task) == list:
                for t in task:
                    print(f'Agent {agent._id} : {t.pick_id} -> {t.drop_id}')
            else:
                print(f'Agent {agent._id} : {task.pick_id} -> {task.drop_id}')


    def generate_agent_paths(self):
        """generate path/trajectory for each agent based on assignment
        self._agent_paths is of the form:
        { # Dictionary of agent ids and paths
            <agent._id> :   [ # List of this agent's paths
                                [ # List of x-y coordinates
                                    [ <x coord>, <y coord> ]
                                ]
                            ]
            <agent._id> :   [ # List of this agent's paths
                                [ # List of x-y coordinates
                                    [ <x coord>, <y coord> ]
                                ]
                            ]
        }
        Access as self._agent_paths[<agent._id>][<path #>][<path (list) #>][<path point #>][<path point coord # (0 for x, 1 for y)>]
        """
        # your code here...
        print("Generating agent paths...")
        self.room_map.disp_grid_xy(self.room_map.o_map)
        for agent, task in self._task_assignment.items():
            this_agents_paths_2D = []
            if type(task) == list:
                last_coords = [agent.get_pos().x, agent.get_pos().y]
                for i in range(len(task)):
                    start_coords = last_coords
                    pick_coords = [task[i].pick_loc.x, task[i].pick_loc.y]
                    drop_coords = [task[i].drop_loc.x, task[i].drop_loc.y]
                    last_coords = drop_coords
                    print(f"Finding path part 1 for {agent._id} task #{i} from {start_coords} to {pick_coords}...")
                    this_agents_paths_2D.append(
                        find_path(start_coords, pick_coords, self.room_map)
                        )
                    print(f"Finding path part 2 for {agent._id} task #{i} from {pick_coords} to {drop_coords}...")
                    this_agents_paths_2D.append(
                        find_path(pick_coords, drop_coords, self.room_map)
                        )
                    self._agent_paths[agent._id] = this_agents_paths_2D
            else:
                start_coords = [agent.get_pos().x, agent.get_pos().y]
                pick_coords = [task.pick_loc.x, task.pick_loc.y]
                drop_coords = [task.drop_loc.x, task.drop_loc.y]
                print(f"Finding path part 1 for {agent._id} from {start_coords} to {pick_coords}...")
                this_agents_paths_2D.append(
                    find_path(start_coords, pick_coords, self.room_map)
                    )
                print(f"Finding path part 2 for {agent._id} from {pick_coords} to {drop_coords}...")
                this_agents_paths_2D.append(
                    find_path(pick_coords, drop_coords, self.room_map)
                )
                self._agent_paths[agent._id] = this_agents_paths_2D
        print(self._agent_paths)


    def smoothen_paths(self):
        """applies a b-spline to the waypoints to obtain a continous path

        Credits: Algobotics (https://www.youtube.com/watch?v=ueUgHvUT2Z0)        
        """

        # if you need this...

        pass


