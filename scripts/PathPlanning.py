import numpy as np

class RoomMap:
    def __init__(self, grid_range_x, grid_range_y,
                 obst_ranges=None, obst_nodes=None,
                 num_nodes_x=21, num_nodes_y=15):
        """
        Make a new RoomMap object containing all necessary information about area that the crazyflie will use and the grid A* will use for path planning.
        Args:
            grid_range_x (2 element list of floats): Starting and ending x coordinates of the room (m)
            grid_range_y (2 element list of floats): Starting and ending y coordinates of the room (m)
            obst_ranges (3D list): Each top level list element contains coordinates of a rectangular obstacle. Each obstacle is denoted as [[<left side x coords>, <right side x coords>], [<bottom side y coords>, <top side y coords>] (looking down into the 2D x-y plane, same units as grid ranges)
            num_nodes_x (int): Number of nodes to split the room into in the x direction.
            num_nodes_y (int): Number of nodes to split the room into in the y direction.
        """
        self.grid_range_x = grid_range_x
        self.grid_range_y = grid_range_y
        self.obst_ranges = obst_ranges
        self.num_nodes_x = num_nodes_x
        self.num_nodes_y = num_nodes_y
        self.x_max = self.num_nodes_x-1
        self.y_max = self.num_nodes_y-1

        self.grid_total_x = self.grid_range_x[1] - self.grid_range_x[0]
        self.grid_total_y = self.grid_range_y[1] - self.grid_range_y[0]
        self.square_width = self.grid_total_x/(self.num_nodes_x-1)
        self.square_height = self.grid_total_y/(self.num_nodes_y-1)

        self.blocked_marker = 0
        self.free_marker = 1
        self.obst_ranges = obst_ranges
        self.obst_nodes = obst_nodes
        if self.obst_ranges != None:
            self.o_map = self.make_o_map(self.num_nodes_x, self.num_nodes_y,
                self.obst_ranges, make_walls=True, free_marker=self.free_marker,
                blocked_marker=self.blocked_marker, datatype=int)
            self.disp_map = self.make_o_map(self.num_nodes_x, self.num_nodes_y,
                self.obst_ranges, make_walls=True, free_marker=self.free_marker,
                blocked_marker=self.blocked_marker, datatype=int)
        if self.obst_nodes != None:
            self.o_map = self.make_o_map_from_node_list(
                self.num_nodes_x, self.num_nodes_y, self.obst_nodes,
                make_walls=True, free_marker=self.free_marker,
                blocked_marker=self.blocked_marker, datatype=int)
            self.disp_map = self.make_o_map_from_node_list(
                self.num_nodes_x, self.num_nodes_y, self.obst_nodes,
                make_walls=True, free_marker=self.free_marker,
                blocked_marker=self.blocked_marker, datatype=int)

    def node2coords(self, node_xy):
        """
        Takes [<node x>, <node y>] and returns [<x coord>, <y coord>].
        """
        x = node_xy[0]*self.square_width + self.grid_range_x[0]
        y = node_xy[1]*self.square_height + self.grid_range_y[0]
        return [x, y]

    def coords2node(self, coords_xy):
        """
        Takes [<x coord>, <y coord>] and returns [<node x>, <node y>].
        """
        x = (coords_xy[0] - self.grid_range_x[0])/self.square_width
        y = (coords_xy[1] - self.grid_range_y[0])/self.square_height
        return [round(x), round(y)]
        
    def mark_walls(self, node_grid):
        """
        Takes a numpy int array of nodes and returns the array with the blocked marker on each node on an edge.
        """
        node_grid[:, 0] = self.blocked_marker
        node_grid[:, self.num_nodes_y-1] = self.blocked_marker
        node_grid[0] = self.blocked_marker
        node_grid[self.num_nodes_x-1] = self.blocked_marker
        return node_grid

    def mark_obst(self, node_grid, obst_ranges):
        """
        Takes a numpy int array of nodes and obstacle coordinates (see self.obst_ranges) and returns the array with the blocked marker on each node in those ranges.
        """
        cn1 = self.coords2node([obst_ranges[0][0],obst_ranges[1][0]])
        cn2 = self.coords2node([obst_ranges[0][1],obst_ranges[1][1]])
        node_grid[cn1[0]:cn2[0]+1, cn1[1]:cn2[1]+1] = self.blocked_marker
        return node_grid

    def mark_node(self, node_grid, mark_coords, mark_val):
        """
        Returns the node grid with a given point [<x in m>, <y in m>]
        """
        x, y = self.coords2node(mark_coords)
        node_grid[x, y] = mark_val
        return node_grid

    def make_o_map(self, num_nodes_x, num_nodes_y, obst_ranges, make_walls=True, free_marker=1, blocked_marker=0, datatype=int):
        """
        Makes the obstacle map of nodes.
        """
        node_grid = np.full([num_nodes_x, num_nodes_y], free_marker, dtype=datatype)
        if make_walls:
            node_grid = self.mark_walls(node_grid)
        for obst_range in obst_ranges:
            node_grid = self.mark_obst(node_grid, obst_range)
        return node_grid

    def make_o_map_from_node_list(self, num_nodes_x, num_nodes_y, obst_nodes, make_walls=True, free_marker=1, blocked_marker=0, datatype=int):
        """
        Makes the obstacle map of nodes.
        """
        node_grid = np.full([num_nodes_x, num_nodes_y], free_marker, dtype=datatype)
        if make_walls:
            node_grid = self.mark_walls(node_grid)
        for obst_node in obst_nodes:
            obst_x_m = obst_node[0]*0.1
            obst_y_m = obst_node[1]*0.1
            node_rc = self.coords2node([obst_x_m, obst_y_m])
            node_grid[node_rc[0], node_rc[1]] = blocked_marker
        return node_grid

    def node_blocked(self, node_x, node_y):
        if ( node_x > self.x_max or node_x < 0 or 
             node_y > self.y_max or node_y < 0 ):
            return True
        else:
            if self.o_map[node_x, node_y] == self.blocked_marker:
                return True
            else:
                return False

    def disp_grid_xy(self, node_grid, show_labels=True):
        """
        Displays a node grid with row (node-x) and column (node-y) numbers.
        """
        arr = np.flip(np.transpose(node_grid), axis=0)
        if show_labels == False:
            print(arr)
        else:
            for i in range(len(arr)):
                row_num = len(arr)-1-i
                if row_num < 10:
                    print(row_num, " |", arr[i])
                else:
                    print(row_num, "|", arr[i])
            x_labels_0 = "     -"
            x_labels_1 = "      "
            x_labels_2 = "      "
            for i in range(len(arr[0])):
                x_labels_0 += "--"
                if i < 10:
                    x_labels_1 += str(i) + " "
                    x_labels_2 += "  "
                else:
                    x_labels_1 += str(i//10) + " "
                    x_labels_2 += str(i % 10) + " "
            print(x_labels_0)
            print(x_labels_1)
            print(x_labels_2)


class Node:
    """
    Attributes:
        x (int): Node's row
        y (int): Node's column
        xy (2 element list): Node's row and column
        gn (2 element list): This path's goal node.
        children (list of Node objects): The best adjacent nodes to this node
        parents (list of Node objects): The chain of nodes that created this one, in chronological order.
        parents_list (list [row,col] pairs): The x and y of nodes in parents.
        cost (float): Cost of reaching this node through parent path
        dist (float): Manhattan distance to the goal node
        f (float): Total node value (cost + dist), lower is better.
        is_blocked (bool): Whether this node is in an obstacle or wall.
    """

    def __init__(self, gn, node_xy, parent):
        """
        Defines a new Node object.

        Args:
            gn (2 element list): Goal node (for this path segment) described as [<node row/x>, <node col/y>]
            node_xy (2 element list): This node's row/x and col/y as [<node row/x>, <node col/y>]
            parent (Node object): The node object this Node was created by. None if this is the start node.
        """
        self.gn = gn
        self.xy = node_xy
        self.x = node_xy[0]
        self.y = node_xy[1]

        self.children = []
        self.parents_list = []

        # Proceedure if start node
        if parent == None:
            self.parents = []
            self.cost = 0
        # If not start node, include parent information
        else:
            # Add parent to list
            self.parents = parent.parents + [parent]
            # Calculate cost as parent's cost plus 1
            self.cost = parent.cost + 1
            # If a corner was turned, add 0.1 to cost to show less desireable
            if len(self.parents) > 1:
                if self.parents[-2].x != self.x and self.parents[-2].y != self.y:
                    self.cost += 0.1
            # Create parents_list for easy printing and path retrieval
            for par in self.parents:
                self.parents_list.append(par.xy)

        # Calculate Manhattan distance to goal node
        self.dist = abs(gn[0]-self.x) + abs(gn[1]-self.y)
        # Calculate f as dist + cost
        self.f = self.dist + self.cost

    def __repr__(self):
        children_list = []
        for chi in self.children:
            children_list.append((chi.x, chi.y))
        return f"Node([{self.x}, {self.y}], f={self.f}, dist={self.dist}, "\
                f"cost={self.cost}, " \
                f"parents={self.parents_list}, children={children_list})"

    def __lt__(self, other):
        """
        Define self.f as the sorting attribute of the Node class.
        """
        return self.f < other.f

    def calc_dist_and_f(self):
        """
        Recalculate distance and f values.
        """
        self.dist = abs(gn[0]-self.x) + abs(gn[1]-self.y)
        self.f = self.dist + self.cost

    def add_children(self, child_nodes):
        """
        Add list of child_nodes to self.children.
        """
        self.children.extend(child_nodes)

    def sort_children(self):
        """
        Sort the list of children by their f values.
        """
        self.children.sort()

    def drop_children(self, keep_num=3):
        """
        Drop all but the best 2 children from this nodes children list.
        """
        if len(self.children) > keep_num:
            self.children = self.children[0:keep_num]

def add_nodes_valid_children(node, m):
    # possible_children_list = [Node(node.gn, [node.x+1, node.y], node),
    #                    Node(node.gn, [node.x-1, node.y], node),
    #                    Node(node.gn, [node.x, node.y+1], node),
    #                    Node(node.gn, [node.x, node.y-1], node)]
    possible_children_list = [Node(node.gn, [node.x, node.y-1], node),
                       Node(node.gn, [node.x, node.y+1], node),
                       Node(node.gn, [node.x+1, node.y], node),
                       Node(node.gn, [node.x-1, node.y], node)]
    valid_children = []
    for c in possible_children_list: # Only take unblocked children
        if m.node_blocked(c.x, c.y) == False:
            valid_children.append(c)
        node.add_children(valid_children)
        node.sort_children()
        node.drop_children()

def print_node_list(node_list):
    """
    Print a list of nodes on separate lines.
    """
    print("[")
    for node in node_list:
        string = f"({node.x},{node.y}) "
        string += str(node.parents_list)
        print("  ", string, ";")
    print("]")

def simplify_path(path):
    """
    Remove all nodes from a path that do not indicate a change in direction or the start or end of the path.
    """
    if len(path) < 3:
        return path
    new_path = [path[0]]
    for i in range(2, len(path)):
        if path[i][0] != path[i-2][0] and path[i][1] != path[i-2][1]:
            new_path.append(path[i-1])
    new_path.append(path[-1])
    return new_path

def convert_path(path, m, round_dig=2):
    """
    Returns the x-y coordinates of each node in the path.
    """
    coord_path = []
    for n in path:
        coords = m.node2coords(n)
        coord_path.append([round(coords[0], round_dig), round(coords[1], round_dig)])
    return coord_path

def find_path(start_coords, goal_coords, m, use_exact_inputs=True, simplify_path=False, print_options=False, max_depth=35):
    """
    Uses A* to find the shortest path from start_coords to goal_coords without hitting any walls or obstacles defined by RoomMap m.
    """
    
    # Get start and goal nodes
    start_node_xy = m.coords2node(start_coords)
    goal_node_xy = m.coords2node(goal_coords)
    
    # Create start node and find it's children
    start = Node(goal_node_xy, start_node_xy,None)
    add_nodes_valid_children(start, m)
    
    # Continuously find the children of each node's best children until at least 5 children reach the goal or the path is max_depth children deep.
    max_cans = 50
    sucessful_children = []
    candidates = [start]
    print("Current depth: ", end='')
    for i in range(max_depth):
        print(i, end=' ')
        new_candidates = []
        for can in candidates:
            if can.xy == goal_node_xy:
                sucessful_children.append(can)
            else:
                add_nodes_valid_children(can, m)
                new_candidates.extend(can.children)
        candidates = new_candidates
        candidates.sort()
        # print("CANDIDATES: ", candidates)
        if len(candidates) > max_cans:
            candidates = candidates[:max_cans]
        if len(sucessful_children) > 5:
            break
    print()

    # Sort children that reached the goal by their f value (dist=0 so f=cost)
    sucessful_children.sort()
    if print_options:
        for c in sucessful_children:
            print([c.parents_list, c.xy, c.f])

    # If at least one path is found, simplify and convert the best path to coordinates and return the converted path as a list of [x,y] points.
    if len(sucessful_children) > 0 :
        path = sucessful_children[0].parents_list
        path.append(sucessful_children[0].xy)
        print(path)
        if simplify_path:
            path = simplify_path(path)
        converted_path = convert_path(path, m)
        if use_exact_inputs:
            converted_path.insert(0, start_coords)
            converted_path.append(goal_coords)
        print(converted_path)
        return converted_path
    # If no path is found, return an empty list
    else:
        return []
