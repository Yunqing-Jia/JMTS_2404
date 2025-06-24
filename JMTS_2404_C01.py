import tkinter as tk
from tkinter import ttk, messagebox, PhotoImage
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import math
from matplotlib.patches import FancyArrowPatch


class DijkstraGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Network Shortest Path")
        self.root.geometry("1200x700")

        # Initialize variables
        self.num_nodes = 0
        self.nodes_dict = {}
        self.distance_matrix = None
        self.graph = nx.DiGraph()  # Use directed graph
        self.shortest_path = []
        self.node_radius = 0.3  # Radius for node collision detection

        # Create GUI
        self.create_widgets()

    def generate_node_name(self, index):
        """Generate node name: A, B, ..., Z, AA, AB, ..."""
        if index < 26:
            return chr(ord('A') + index)
        else:
            first = chr(ord('A') + (index - 26) // 26)
            second = chr(ord('A') + (index - 26) % 26)
            return first + second

    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for problem definition
        left_frame = ttk.LabelFrame(main_frame, text="Problem Definition", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Right panel for visualization
        right_frame = ttk.LabelFrame(main_frame, text="Network Visualization", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # === LEFT PANEL CONTENT ===
        # Row 1: Number of nodes input
        row1_frame = ttk.Frame(left_frame)
        row1_frame.pack(fill=tk.X, pady=5)

        ttk.Label(row1_frame, text="Number of nodes (≥2):").pack(side=tk.LEFT)
        self.num_nodes_var = tk.StringVar(value="5")
        ttk.Entry(row1_frame, textvariable=self.num_nodes_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(row1_frame, text="Define Points", command=self.define_points).pack(side=tk.LEFT, padx=5)

        # Row 2: Node information table
        row2_frame = ttk.LabelFrame(left_frame, text="Node Information", padding="5")
        row2_frame.pack(fill=tk.X, pady=10)

        # Create frame for node table with headers
        table_frame = ttk.Frame(row2_frame)
        table_frame.pack(fill=tk.X, expand=True)

        # Create headers
        ttk.Label(table_frame, text="ID", font=('Arial', 9, 'bold')).grid(row=0, column=0, padx=5, pady=2, sticky='w')
        ttk.Label(table_frame, text="Name", font=('Arial', 9, 'bold')).grid(row=0, column=1, padx=5, pady=2, sticky='w')
        ttk.Label(table_frame, text="X", font=('Arial', 9, 'bold')).grid(row=0, column=2, padx=5, pady=2, sticky='w')
        ttk.Label(table_frame, text="Y", font=('Arial', 9, 'bold')).grid(row=0, column=3, padx=5, pady=2, sticky='w')

        # Create scrollable frame for node entries
        node_canvas = tk.Canvas(table_frame, height=150)
        node_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=node_canvas.yview)
        self.node_entries_frame = ttk.Frame(node_canvas)

        node_canvas.configure(yscrollcommand=node_scrollbar.set)
        node_canvas.grid(row=1, column=0, columnspan=4, sticky='nsew', pady=5)
        node_scrollbar.grid(row=1, column=4, sticky='ns', pady=5)

        node_canvas.create_window((0, 0), window=self.node_entries_frame, anchor='nw')

        # Configure grid weights
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_columnconfigure(1, weight=1)
        table_frame.grid_columnconfigure(2, weight=1)
        table_frame.grid_columnconfigure(3, weight=1)

        self.node_canvas = node_canvas
        self.node_entries = {}

        # Row 3: Distance matrix
        row3_frame = ttk.LabelFrame(left_frame, text="Distance Matrix", padding="5")
        row3_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create main container for matrix
        matrix_main_frame = ttk.Frame(row3_frame)
        matrix_main_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas and scrollbars for matrix
        self.matrix_canvas = tk.Canvas(matrix_main_frame, bg='white')
        v_scrollbar = ttk.Scrollbar(matrix_main_frame, orient=tk.VERTICAL, command=self.matrix_canvas.yview)
        h_scrollbar = ttk.Scrollbar(matrix_main_frame, orient=tk.HORIZONTAL, command=self.matrix_canvas.xview)

        self.matrix_frame = ttk.Frame(self.matrix_canvas)

        # Configure scrollbars
        self.matrix_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.matrix_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create window in canvas
        self.matrix_window = self.matrix_canvas.create_window((0, 0), window=self.matrix_frame, anchor='nw')

        # Bind canvas events
        self.matrix_frame.bind('<Configure>', self._on_matrix_frame_configure)
        self.matrix_canvas.bind('<Configure>', self._on_matrix_canvas_configure)

        row4_frame = ttk.Frame(left_frame)
        row4_frame.pack(fill=tk.X, pady=10)

        img = PhotoImage(file='docs/J-MTS.png')
        img_label = tk.Label(row4_frame, image=img)
        img_label.image = img
        img_label.pack(side='left')

        ttk.Button(row4_frame, text="Define Network", command=self.define_network).pack(side='right')

        # === RIGHT PANEL CONTENT ===
        # Control row for shortest path calculation
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Start:").pack(side=tk.LEFT)
        self.start_var = tk.StringVar()
        self.start_combo = ttk.Combobox(control_frame, textvariable=self.start_var, width=8)
        self.start_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="End:").pack(side=tk.LEFT, padx=(10, 0))
        self.end_var = tk.StringVar()
        self.end_combo = ttk.Combobox(control_frame, textvariable=self.end_var, width=8)
        self.end_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Algorithm:").pack(side=tk.LEFT, padx=(10, 0))
        self.algorithm_var = tk.StringVar(value="Dijkstra")
        algorithm_combo = ttk.Combobox(control_frame, textvariable=self.algorithm_var,
                                       values=["Dijkstra", "BFS"], width=12)
        algorithm_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Calculate Shortest Path",
                   command=self.calculate_shortest_path).pack(side=tk.LEFT, padx=10)

        # Visualization area
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize with default
        self.define_points()

    def define_points(self):
        """Define points based on input number"""
        try:
            num = int(self.num_nodes_var.get())
            if num < 2:
                messagebox.showerror("Error", "Number of nodes must be ≥ 2")
                return

            self.num_nodes = num

            # Generate nodes dictionary
            self.nodes_dict = {}
            for i in range(num):
                node_id = self.generate_node_name(i)
                self.nodes_dict[node_id] = {
                    'name': node_id,
                    'x': i + 1,
                    'y': i + 1
                }

            # Generate distance matrix (n×n with inf except diagonal)
            self.distance_matrix = np.full((num, num), np.inf)
            np.fill_diagonal(self.distance_matrix, 0)

            # Update displays
            self.update_node_table()
            self.create_matrix_widgets()
            self.update_combos()
            self.draw_network()

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")

    def _on_matrix_frame_configure(self, event):
        """Update canvas scroll region when frame size changes"""
        self.matrix_canvas.configure(scrollregion=self.matrix_canvas.bbox("all"))

    def _on_matrix_canvas_configure(self, event):
        """Update canvas window size when canvas size changes"""
        canvas_width = event.width
        self.matrix_canvas.itemconfig(self.matrix_window, width=canvas_width)

    def update_node_table(self):
        """Update the node information table"""
        # Clear existing entries
        for widget in self.node_entries_frame.winfo_children():
            widget.destroy()

        self.node_entries = {}

        # Add new entries
        row = 0
        for node_id, info in self.nodes_dict.items():
            # ID label (non-editable)
            ttk.Label(self.node_entries_frame, text=node_id).grid(row=row, column=0, padx=5, pady=2, sticky='w')

            # Name entry
            name_var = tk.StringVar(value=info['name'])
            name_entry = ttk.Entry(self.node_entries_frame, textvariable=name_var, width=10)
            name_entry.grid(row=row, column=1, padx=5, pady=2, sticky='w')
            name_entry.bind('<FocusOut>', lambda e, nid=node_id: self.update_node_name(nid, e.widget.get()))

            # X coordinate entry
            x_var = tk.StringVar(value=str(info['x']))
            x_entry = ttk.Entry(self.node_entries_frame, textvariable=x_var, width=8)
            x_entry.grid(row=row, column=2, padx=5, pady=2, sticky='w')
            x_entry.bind('<FocusOut>', lambda e, nid=node_id: self.update_node_x(nid, e.widget.get()))

            # Y coordinate entry
            y_var = tk.StringVar(value=str(info['y']))
            y_entry = ttk.Entry(self.node_entries_frame, textvariable=y_var, width=8)
            y_entry.grid(row=row, column=3, padx=5, pady=2, sticky='w')
            y_entry.bind('<FocusOut>', lambda e, nid=node_id: self.update_node_y(nid, e.widget.get()))

            self.node_entries[node_id] = {
                'name': name_entry,
                'x': x_entry,
                'y': y_entry
            }

            row += 1

        # Update scroll region
        self.node_entries_frame.update_idletasks()
        self.node_canvas.configure(scrollregion=self.node_canvas.bbox("all"))

    def update_node_name(self, node_id, new_name):
        """Update node name"""
        if new_name.strip():
            self.nodes_dict[node_id]['name'] = new_name.strip()
            self.update_combos()

    def update_node_x(self, node_id, new_x):
        """Update node X coordinate"""
        try:
            x_val = float(new_x)
            self.nodes_dict[node_id]['x'] = x_val
            self.draw_network()
        except ValueError:
            # Reset to original value
            self.node_entries[node_id]['x'].delete(0, tk.END)
            self.node_entries[node_id]['x'].insert(0, str(self.nodes_dict[node_id]['x']))

    def update_node_y(self, node_id, new_y):
        """Update node Y coordinate"""
        try:
            y_val = float(new_y)
            self.nodes_dict[node_id]['y'] = y_val
            self.draw_network()
        except ValueError:
            # Reset to original value
            self.node_entries[node_id]['y'].delete(0, tk.END)
            self.node_entries[node_id]['y'].insert(0, str(self.nodes_dict[node_id]['y']))

    def create_matrix_widgets(self):
        """Create distance matrix input widgets"""
        # Clear existing widgets
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()

        if self.num_nodes == 0:
            return

        node_ids = list(self.nodes_dict.keys())

        # Create two-line label for origin\destination
        origin_dest_frame = ttk.Frame(self.matrix_frame)
        origin_dest_frame.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')

        # First line: destination→
        dest_label = ttk.Label(origin_dest_frame, text="destination→",
                               font=('Arial', 8, 'bold'), justify='center')
        dest_label.pack()

        # Second line: origin↓
        origin_label = ttk.Label(origin_dest_frame, text="origin↓",
                                 font=('Arial', 8, 'bold'), justify='center')
        origin_label.pack()

        # Column headers
        for j, node_id in enumerate(node_ids):
            header_label = ttk.Label(self.matrix_frame, text=node_id,
                                     font=('Arial', 9, 'bold'))
            header_label.grid(row=0, column=j + 1, padx=2, pady=2, sticky='nsew')

        # Row headers and matrix entries
        self.matrix_entries = {}
        for i, from_node in enumerate(node_ids):
            # Row header
            row_label = ttk.Label(self.matrix_frame, text=from_node,
                                  font=('Arial', 9, 'bold'))
            row_label.grid(row=i + 1, column=0, padx=2, pady=2, sticky='nsew')

            self.matrix_entries[i] = {}
            for j, to_node in enumerate(node_ids):
                if i == j:  # Diagonal elements
                    entry = ttk.Entry(self.matrix_frame, width=6, state='readonly',
                                      justify='center')
                    entry.grid(row=i + 1, column=j + 1, padx=1, pady=1, sticky='nsew')
                    entry.config(state='normal')
                    entry.delete(0, tk.END)
                    entry.insert(0, "0")
                    entry.config(state='readonly')
                else:
                    entry = ttk.Entry(self.matrix_frame, width=6, justify='center')
                    entry.grid(row=i + 1, column=j + 1, padx=1, pady=1, sticky='nsew')

                    # Set initial value
                    if np.isinf(self.distance_matrix[i][j]):
                        entry.insert(0, "inf")
                    else:
                        entry.insert(0, str(self.distance_matrix[i][j]))

                    # Bind event to update matrix when value changes
                    entry.bind('<FocusOut>', lambda e, r=i, c=j: self.update_matrix_value(r, c))
                    entry.bind('<Return>', lambda e, r=i, c=j: self.update_matrix_value(r, c))

                self.matrix_entries[i][j] = entry

        # Configure grid weights for better layout
        for i in range(len(node_ids) + 1):
            self.matrix_frame.grid_columnconfigure(i, weight=1)
            if i > 0:
                self.matrix_frame.grid_rowconfigure(i, weight=1)

        # Update scroll region after creating widgets
        self.matrix_frame.update_idletasks()
        self.matrix_canvas.configure(scrollregion=self.matrix_canvas.bbox("all"))

    def update_matrix_value(self, row, col):
        """Update distance matrix value"""
        try:
            entry = self.matrix_entries[row][col]
            value_str = entry.get().strip()

            if value_str.lower() == 'inf' or value_str == '':
                self.distance_matrix[row][col] = np.inf
                entry.delete(0, tk.END)
                entry.insert(0, "inf")
            else:
                value = float(value_str)
                if value <= 0:
                    messagebox.showerror("Error", "Distance must be positive")
                    entry.delete(0, tk.END)
                    entry.insert(0, "inf")
                    self.distance_matrix[row][col] = np.inf
                else:
                    self.distance_matrix[row][col] = value
                    entry.delete(0, tk.END)
                    entry.insert(0, str(value))

            # Update visualization
            self.draw_network()

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number or 'inf'")
            entry = self.matrix_entries[row][col]
            entry.delete(0, tk.END)
            entry.insert(0, "inf")
            self.distance_matrix[row][col] = np.inf

    def define_network(self):
        """Update network visualization"""
        self.draw_network()

    def update_combos(self):
        """Update start and end combo boxes"""
        node_names = [info['name'] for info in self.nodes_dict.values()]
        self.start_combo['values'] = node_names
        self.end_combo['values'] = node_names
        if node_names:
            self.start_var.set(node_names[0])
            if len(node_names) > 1:
                self.end_var.set(node_names[1])

    def calculate_edge_endpoints(self, pos1, pos2):
        """Calculate edge start and end points that don't overlap with nodes"""
        x1, y1 = pos1
        x2, y2 = pos2

        # Calculate direction vector
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx ** 2 + dy ** 2)

        if length == 0:
            return pos1, pos2

        # Normalize direction vector
        dx_norm = dx / length
        dy_norm = dy / length

        # Calculate start and end points offset by node radius
        start_x = x1 + dx_norm * self.node_radius
        start_y = y1 + dy_norm * self.node_radius
        end_x = x2 - dx_norm * self.node_radius
        end_y = y2 - dy_norm * self.node_radius

        return (start_x, start_y), (end_x, end_y)

    def draw_curved_edge(self, pos1, pos2, ax, color='black', width=1, is_shortest_path=False):
        """Draw a curved edge between two positions with improved positioning"""
        # Calculate edge endpoints that don't overlap with nodes
        start_pos, end_pos = self.calculate_edge_endpoints(pos1, pos2)

        # Create curved arrow with better curvature
        style = "arc3,rad=0.2"  # Reduced curvature for better visibility
        arrow = FancyArrowPatch(start_pos, end_pos,
                                connectionstyle=style,
                                arrowstyle='->',
                                mutation_scale=15,
                                color=color,
                                linewidth=width,
                                zorder=1)  # Lower z-order to draw behind nodes
        ax.add_patch(arrow)

    def calculate_label_position_on_curve(self, pos1, pos2, curve_rad=-0.2):
        """Calculate position for label directly on the curved arrow line with precise midpoint calculation"""
        x1, y1 = pos1
        x2, y2 = pos2

        # First calculate the adjusted endpoints (same as in draw_curved_edge)
        start_pos, end_pos = self.calculate_edge_endpoints(pos1, pos2)
        x1_adj, y1_adj = start_pos
        x2_adj, y2_adj = end_pos

        # Calculate the distance and direction of the adjusted line
        dx = x2_adj - x1_adj
        dy = y2_adj - y1_adj
        length = math.sqrt(dx ** 2 + dy ** 2)

        if length == 0:
            return x1, y1

        # Calculate perpendicular vector for curve control
        perp_x = -dy / length
        perp_y = dx / length

        # Control point offset (same as FancyArrowPatch uses for arc3,rad=0.2)
        control_offset = curve_rad * length

        # Calculate the midpoint of the straight line between adjusted endpoints
        mid_x = (x1_adj + x2_adj) / 2
        mid_y = (y1_adj + y2_adj) / 2

        # Calculate control point for the curve
        control_x = mid_x + perp_x * control_offset
        control_y = mid_y + perp_y * control_offset

        # For arc3 connection style, the actual curve is a quadratic Bezier curve
        # The midpoint of a quadratic Bezier at t=0.5 is:
        # P(0.5) = 0.25*P0 + 0.5*P_control + 0.25*P2
        label_x = 0.25 * x1_adj + 0.5 * control_x + 0.25 * x2_adj
        label_y = 0.25 * y1_adj + 0.5 * control_y + 0.25 * y2_adj

        return label_x, label_y

    def draw_network(self):
        """Draw the network graph with labels positioned precisely on arrow lines"""
        self.ax.clear()

        if not self.nodes_dict:
            self.canvas.draw()
            return

        # Create graph
        self.graph = nx.DiGraph()
        node_ids = list(self.nodes_dict.keys())

        # Add nodes
        pos = {}
        for node_id, info in self.nodes_dict.items():
            self.graph.add_node(node_id)
            pos[node_id] = (info['x'], info['y'])

        # Add edges based on distance matrix
        for i, from_node in enumerate(node_ids):
            for j, to_node in enumerate(node_ids):
                if i != j and not np.isinf(self.distance_matrix[i][j]):
                    self.graph.add_edge(from_node, to_node, weight=self.distance_matrix[i][j])

        # Identify shortest path edges
        shortest_path_edges = set()
        if len(self.shortest_path) > 1:
            for i in range(len(self.shortest_path) - 1):
                shortest_path_edges.add((self.shortest_path[i], self.shortest_path[i + 1]))

        # Draw edges first (so they appear behind nodes)
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        for edge in self.graph.edges():
            from_node, to_node = edge
            pos1 = pos[from_node]
            pos2 = pos[to_node]

            if edge in shortest_path_edges:
                # Draw shortest path edges in blue with thicker line
                self.draw_curved_edge(pos1, pos2, self.ax, color='blue', width=3, is_shortest_path=True)
            else:
                # Draw normal edges in black
                self.draw_curved_edge(pos1, pos2, self.ax, color='black', width=1)

        # Draw edge labels positioned precisely on the curved lines
        for edge, weight in edge_labels.items():
            from_node, to_node = edge
            pos1 = pos[from_node]
            pos2 = pos[to_node]

            # Calculate label position precisely on the curved line
            label_x, label_y = self.calculate_label_position_on_curve(pos1, pos2)

            # Determine label color based on whether it's part of shortest path
            label_color = 'blue' if edge in shortest_path_edges else 'black'

            # Use compact label style with precise positioning
            self.ax.text(label_x, label_y, f'{weight:.1f}',
                         fontsize=8, ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.15',
                                   facecolor='white',
                                   edgecolor='gray',
                                   alpha=0.9,
                                   linewidth=0.5),
                         color=label_color,
                         weight='bold' if edge in shortest_path_edges else 'normal',
                         zorder=3)  # Higher z-order to appear above everything

        # Draw nodes last (so they appear on top of edges)
        nx.draw_networkx_nodes(self.graph, pos, ax=self.ax,
                               node_color='lightblue',
                               node_size=1000,
                               edgecolors='navy',
                               linewidths=2)

        # Draw node labels
        node_labels = {node_id: node_id for node_id in node_ids}
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels,
                                ax=self.ax, font_size=12, font_weight='bold')

        self.ax.set_title("Network Graph", fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)

        # Show x and y ticks
        #self.ax.tick_params(axis='both', which='major', labelsize=8)

        self.ax.tick_params(axis='both', which='major', labelsize=8, labelleft=True, labelbottom=True)

        # Set axis labels
        self.ax.set_xlabel('X coordinate', fontsize=10)
        self.ax.set_ylabel('Y coordinate', fontsize=10)

        # Equal aspect ratio for better visualization
        self.ax.set_aspect('equal', adjustable='box')

        self.canvas.draw()

    def calculate_shortest_path(self):
        """Calculate shortest path using selected algorithm"""
        start_name = self.start_var.get()
        end_name = self.end_var.get()

        if not start_name or not end_name:
            messagebox.showerror("Error", "Please select start and end nodes")
            return

        # Find node IDs by name
        start_id = None
        end_id = None
        for node_id, info in self.nodes_dict.items():
            if info['name'] == start_name:
                start_id = node_id
            if info['name'] == end_name:
                end_id = node_id

        if not start_id or not end_id:
            messagebox.showerror("Error", "Selected nodes not found")
            return

        try:
            if self.algorithm_var.get() == "Dijkstra":
                path = nx.shortest_path(self.graph, start_id, end_id, weight='weight')
                length = nx.shortest_path_length(self.graph, start_id, end_id, weight='weight')
            else:
                path = nx.shortest_path(self.graph, start_id, end_id, weight=None)
                length = nx.shortest_path_length(self.graph, start_id, end_id, weight=None)
            self.shortest_path = path

            # Show result
            path_names = [self.nodes_dict[node]['name'] for node in path]
            result_msg = f"Shortest path from {start_name} to {end_name}:\n"
            result_msg += f"Path: {' → '.join(path_names)}\n"
            result_msg += f"Total distance: {length:.2f}"

            messagebox.showinfo("Shortest Path Result", result_msg)

            # Redraw graph with highlighted path
            self.draw_network()

        except nx.NetworkXNoPath:
            messagebox.showerror("Error", f"No path from {start_name} to {end_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Calculation error: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DijkstraGUI(root)
    root.mainloop()