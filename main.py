from scripts.utils import Task, Position, State, VelCommand
from scripts.Quadrotor import Quadrotor
from scripts.Simulation import Simulation
from scripts.GroundControl import GroundControlSystem
import plotly.graph_objects as go
import yaml
import time

def run():
    
    # #########################################################################################################
    #### STEP 1: Define the task list, agent list and environment parameters
    # #########################################################################################################

    # load configuration file from YAML file
    with open('./scripts/config.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    use_hardware = config['use_hardware']
    agent_init = config['agent_init']
    colors = config['agent_colors']
    time_delta = config['time_delta']
    env = config['map']
    num_agents = len(agent_init)


    print('Starting up the notebook for the Multi-Agent Coordination module... \n')
    print(f'Number of Agents: [{num_agents}] -> {[agent_init[i][0] for i in range(len(agent_init))]}')
    print(f'Use Hardware: [{use_hardware}]')
    print(f'Time Delta (dt): [{time_delta}]')


    # ---------------------------------------------------------------------------------------------------------
    #### STEP 1.A: Define the task list
    # ---------------------------------------------------------------------------------------------------------

    # create the task list
    task_list = dict()
    pick_locations = config['pick_loc']
    drop_locations = config['drop_loc']

    print('-----------------------')
    print('Task List')
    print('-----------------------')

    for i in range(len(pick_locations)):
        t = Task(pick_loc=Position(x=pick_locations[i][1][0], y=pick_locations[i][1][1], z=pick_locations[i][1][2]),
                drop_loc=Position(x=drop_locations[i][1][0], y=drop_locations[i][1][1], z=0.0), 
                pick_id=pick_locations[i][0], 
                drop_id=drop_locations[i][0], 
                id='T'+str(i), 
                priority=i)
        task_list[pick_locations[i][0]] = t

        print(f'Task {t.id}: {t.pick_id} -> {t.drop_id}')

    print('-----------------------')


    # ---------------------------------------------------------------------------------------------------------
    #### STEP 1.B: Define the agent list
    # ---------------------------------------------------------------------------------------------------------

    print('----------------------------------')
    print('Agent List (with home position)')
    print('----------------------------------')

    agent_list = dict()
    for i in range(num_agents):
        # define initial state
        start = State(x_pos=agent_init[i][1][0], 
                    y_pos=agent_init[i][1][1])
        # define the appropriate UR1
        uri = 'radio://0/'+agent_init[i][0][2:]+'0/2M/E7E7E7E7E7'
        # define agent as Quadrotor
        agent = Quadrotor(init_state=start, 
                        color=colors[i], 
                        id=agent_init[i][0], 
                        uri=uri,
                        take_off_height=agent_init[i][2], 
                        hardware_flag=use_hardware,
                        dt=time_delta)
        agent_list[agent._id] = agent

        if use_hardware:
            print(f'Agent {agent._id}: {agent_list[agent._id].get_pos().x, agent_list[agent._id].get_pos().y} \
                ---> {uri}')
        else:
            print(f'Agent {agent._id}: {agent_list[agent._id].get_pos().x, agent_list[agent._id].get_pos().y} ')


    print('----------------------------------')

    if use_hardware:
        print('\n !!!!!!!!Please ensure you confirm the Crazyflies are connected to the right radio channels!!!!!!!!')



    # #########################################################################################################
    #### STEP 2: Implement Multi-Agent Task Assignment
    # #########################################################################################################


    # ---------------------------------------------------------------------------------------------------------
    #### STEP 2.A. Define the Ground Constrol System & compute assignment
    # ---------------------------------------------------------------------------------------------------------

    # instantiate ground control system object
    gcs = GroundControlSystem(agent_list=agent_list, 
                            task_list=task_list,
                            env=env)

    # creates a directed graph based on the agents and task list
    gcs.set_task_graph(draw=True) # toggle draw True or False


    print('----------------------')
    print('Task Assignment')
    print('----------------------')

    # create task assignment
    gcs.create_task_assignment()

    print('----------------------')


    # observe task assignment
    task_assignment = gcs.get_task_assignment(draw=True)



    # #########################################################################################################
    #### STEP 3: Implement Multi-Agent Path Finding
    # #########################################################################################################


    fig1 = go.Figure()
    fig2 = go.Figure()

    sim = Simulation(env=env, fig1=fig1, fig2=fig2)

    sim.add_agents(agent_list)
    sim.set_task_list(task_list)
    sim.init_plot()


    # ---------------------------------------------------------------------------------------------------------
    #### STEP 3.A. Compute collision-free paths
    # ---------------------------------------------------------------------------------------------------------


    # Here, you should implement your multi-agent path finding algorithm...
    gcs.generate_agent_paths()

    # Plotter
    # sim.init_plot()



    # #########################################################################################################
    #### MAIN CONTROL LOOP
    # #########################################################################################################

    # #########################################################################################################
    #### STEP 4: Implement the Path planning & collision avoidance algorithm
    # #########################################################################################################

    # Upload your functions from Module 2 here

    def upload_paths():
        paths = gcs.get_agent_paths()
        for agent in agent_list.values():
            agent.set_trajectory([point for seq in paths[agent._id] for point in seq])
    
    def agent_nav(self):
        V = VelCommand()
        setpoint = self._trajectory[self._traj_index]
        V.vx = 0.2 * (abs(setpoint[0] - self._state.x_pos) > 0)
        V.vy = 0.2 * (abs(setpoint[1] - self._state.y_pos) > 0)
        V.vz = 0.0
        V.v_psi = 0.0
        
        if self._hardware_flag:
            self.velocity_setpoint_hw(V)
        else:
            self.velocity_setpoint_sim(V)


    # #########################################################################################################
    #### STEP 5: Run the Main control loop
    # #########################################################################################################

    time_lapse = 0
    flight_duration = 5


    for agent in agent_list.values():
        agent.initialize_agent()

    time.sleep(3)

    # while True:
    while time_lapse < flight_duration: #secs

        for agent in agent_list.values():

            agent.control_method = agent_nav(agent)
            
            # print out current position of each agent
            x, y, z = agent.get_pos().x, agent.get_pos().y, agent.get_pos().z
            print(f'Agent [{agent._id}]: t = {time_lapse} -> [x, y, z] = [{x:0.3f}, {y:0.3f}, {z:0.3f}]')
                

        # keep track of time lapsed
        time_lapse += time_delta

        time.sleep(time_delta) #TODO: Find what the max delay we can have to good performance with position or velocity command


    for agent in agent_list.values():
        agent.land()


    sim.update_plot()




if __name__ == "__main__":
    run()