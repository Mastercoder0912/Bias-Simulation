import tkinter as tk
import random
import math

#copilot AI was used in the creation of this project helping to guide me into using tkinter, seeded randomness, tension model class, and one-lined if statements.
#it was only used to give me examples of how to use those tools but the rest of the code was written entirely by me.

class TensionModel:
    def __init__(self, seed=12345, alpha=0.85, gamma_seed=0.3, sigma_noise=0.2,
                 beta_c=0.5, beta_d=0.4, beta_a=0.1):
        """
        seed        : base seed for reproducibility
        alpha       : memory coefficient (0.85 = strong inertia, 0.5 = quick swings)
        gamma_seed  : weight of seeded noise vs. structural signals
        sigma_noise : volatility of noise
        beta_c/d/a  : weights for conflict, diversity, anchors (sum ~ 1)
        """
        self.seed = seed
        self.alpha = alpha
        self.gamma_seed = gamma_seed
        self.sigma_noise = sigma_noise
        self.beta_c = beta_c
        self.beta_d = beta_d
        self.beta_a = beta_a
        self.tension = 0.5  # start neutral

    def compute_diversity(self, population):
        # fraction of 1s
        p = sum(node.data for node in population) / len(population)
        return 4 * p * (1 - p)  # max at p=0.5

    def compute_conflict(self, population):
        disagreements, total_edges = 0, 0
        for node in population:
            for neighbor in node.next_nodes:
                total_edges += 1
                if node.data != neighbor.next_nodes:
                    disagreements += 1
            for neighbor in node.prev_nodes:
                total_edges += 1
                if node.data != neighbor.prev_nodes:
                    disagreements += 1
        return disagreements / total_edges if total_edges > 0 else 0

    def compute_anchors(self, population):
        anchors = sum(1 for node in population if getattr(node, "confidence", 0.0) > 0.8)
        return anchors / len(population)

    def update(self, population, step):
        S_conflict = self.compute_conflict(population)
        S_diversity = self.compute_diversity(population)
        S_anchors = self.compute_anchors(population)

        S_struct = (self.beta_c * S_conflict + self.beta_d * S_diversity + self.beta_a * S_anchors)

        rng = random.Random((self.seed << 32) ^ step)
        noise_raw = rng.gauss(0, 1)
        noise = max(-1, min(1, noise_raw * self.sigma_noise))

        T_inst = self.gamma_seed * (0.5 + noise / 2) + (1 - self.gamma_seed) * S_struct
        T_inst = max(0, min(1, T_inst))

        eps = rng.uniform(-1e-4, 1e-4)  # tiny jitter
        self.tension = self.alpha * self.tension + (1 - self.alpha) * T_inst + eps
        self.tension = max(0, min(1, self.tension))

        return self.tension

class Node:
    def __init__(self, data):
        self.data = data
        self.prev_nodes = []
        self.next_nodes = []
        self.next_random_flags = []

    def connect(self, all_nodes, random_connections=0):
        n = len(all_nodes)
        for node in all_nodes:
            node.next_nodes = []
            node.prev_nodes = []
            node.next_random_flags = []

        for i, node in enumerate(all_nodes):
            for j in range(1, 11):
                node.next_nodes.append(all_nodes[(i + j) % n])
                node.next_random_flags.append(False)
                node.prev_nodes.append(all_nodes[(i - j) % n])

        all_edges = [(i, j) for i in range(n) for j in range(10)]
        chosen = random.sample(all_edges, min(random_connections, len(all_edges)))
        for i, j in chosen:
            node = all_nodes[i]
            node.next_nodes[j] = all_nodes[random.randrange(n)]
            node.next_random_flags[j] = True

        return len(chosen)

    def __repr__(self):
        return f"Node({self.data})"


colors = ['red', 'blue']

def bias(population, tense):
    for node in population:
        average = 0
        for person in node.prev_nodes:
            average += person.data
        for person in node.next_nodes:
            average += person.data
        average /= 20

        base_b = 0.5
        b = min(1, max(0, base_b + 0.3 * tense))
        confidence = 1 if node.data == round(average) else 0.5
        exposure = (20 / len(population)) * (1 - tense)
        similarity = 1 - abs(node.data - average)
        similarity_weight = 0.3 + 0.4 * tense

        influence_factor = min(1, max(0, (0.4 * (1 - confidence) + 0.3 * exposure + similarity_weight * (1 - similarity))))

        blended_bias = (1 - b) * average + b * node.data

        adjusted = blended_bias * (1 - influence_factor)
        node.data = int(round(adjusted))

def create(size=20, random_connections=0):
    nodes = [Node(random.randint(0, 1)) for _ in range(size)]
    nodes[0].connect(nodes, random_connections=random_connections)
    return nodes

def visualize(nodes, max_lines=30):
    root = tk.Tk()
    root.title("Bias Simulation")
    root.geometry("1000x820")

    main = tk.Frame(root)
    main.pack(fill="both", expand=True)
    canvas = tk.Canvas(main, width=800, height=800, bg="white")
    canvas.pack(side="left", fill="both", expand=False)
    controls = tk.Frame(main, width=200)
    controls.pack(side="right", fill="y")

    rand_var = tk.IntVar(value=10)
    tk.Label(controls, text="Random connection chance (%)").pack(pady=(10, 5))
    slider = tk.Scale(controls, from_=0, to=100, orient="horizontal", variable=rand_var)
    slider.pack(fill="x", padx=10)

    tk.Label(controls, text="Max lines drawn").pack(pady=(20, 5))
    max_lines_var = tk.IntVar(value=max_lines)
    lines_slider = tk.Scale(controls, from_=5, to=200, orient="horizontal", variable=max_lines_var)
    lines_slider.pack(fill="x", padx=10)

    # Population recreation controls
    tk.Label(controls, text="Population size").pack(pady=(20, 5))
    pop_entry = tk.Entry(controls)
    pop_entry.insert(0, "100")
    pop_entry.pack(padx=10, pady=(0, 5))

    def recreate_population():
        nonlocal nodes
        try:
            size = int(pop_entry.get())
            if size < 2: size = 2
        except ValueError:
            size = 20
        nodes = create(size=size, random_connections=rand_var.get())
        redraw()

    tk.Button(controls, text="Recreate Population", command=recreate_population).pack(pady=(5, 20))

    tk.Label(controls, text="Number of bias iterations").pack(pady=(20, 5))
    runs_entry = tk.Entry(controls)
    runs_entry.insert(0, "1")
    runs_entry.pack(pady=(0, 10))

    def run_bias():
        tension_model = TensionModel()
        try:
            iterations = int(runs_entry.get())
        except ValueError:
            iterations = 10
        for i in range(iterations):
            tense = tension_model.update(nodes, i)
            bias(nodes, tense)
        redraw()

    tk.Button(controls, text="Run Bias", command=run_bias).pack(pady=(5, 20))

    status = tk.Label(controls, text="", anchor="w", justify="left")
    status.pack(fill="x", padx=10, pady=20)

    canvas_size = 800
    radius = len(nodes) * 3
    print(radius)
    center = canvas_size // 2

    def redraw(*_):
        canvas.delete("all")
        random_connections = rand_var.get()
        radius = 350 if (len(nodes) * 3) > 350 else len(nodes) * 3
        nodes[0].connect(nodes, random_connections=random_connections)

        positions = {}
        n = len(nodes)
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / n
            x = center + radius * math.cos(angle)
            y = center + radius * math.sin(angle)
            positions[node] = (x, y)

        max_draw = max_lines_var.get()
        line_count = 0
        red_drawn = 0

        # draw random edges first
        for node in nodes:
            if line_count >= max_draw:
                break
            x1, y1 = positions[node]
            for neighbor, is_rand in zip(node.next_nodes, node.next_random_flags):
                if line_count >= max_draw:
                    break
                if is_rand:
                    x2, y2 = positions[neighbor]
                    canvas.create_line(x1, y1, x2, y2, fill="tomato", width=2)
                    line_count += 1
                    red_drawn += 1

        # fill with gray edges
        if line_count < max_draw:
            for node in nodes:
                if line_count >= max_draw:
                    break
                x1, y1 = positions[node]
                for neighbor, is_rand in zip(node.next_nodes, node.next_random_flags):
                    if line_count >= max_draw:
                        break
                    if not is_rand:
                        x2, y2 = positions[neighbor]
                        canvas.create_line(x1, y1, x2, y2, fill="gray")
                        line_count += 1

        for node in nodes:
            x, y = positions[node]
            r = 8
            color = colors[int(round(node.data)) % len(colors)]
            canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="")
            canvas.create_text(x, y, text=str(round(node.data, 2)), font=("Arial", 7))

        status.config(
            text=f"Nodes: {n}\n"
                 f"Lines drawn: {line_count}\n"
                 f"Random connections: {red_drawn}"
        )

    slider.config(command=redraw)
    lines_slider.config(command=redraw)
    redraw()
    root.mainloop()

population = create(size=100, random_connections=0)

def start_simulation():
    root.destroy()
    visualize(population, max_lines=30)

root = tk.Tk()
root.title("Introduction")

intro_label = tk.Label(root,text="Welcome to my CS50 final project\n This is a bias simulation to show how drastically bias can affect\n different kinds of people \n "
    "This is just a simulation and not real life so results will vary from real life\n\n Red lines represent random connections that people make since most groups\n"
    "know someone who knows someone that is not located close to them. \nGrey lines show local connections. the dots represent people and there colors represent\n their"
    " view on a random topic(I choose red or blue to make it \n easy to understand.)"
    "\n\n\n Click below to begin.", font=("Arial", 14))
intro_label.pack(pady=40)

start_button = tk.Button(root,text="Start Simulation",command=start_simulation,font=("Arial", 12))
start_button.pack(side="bottom", pady=20)
root.mainloop()
