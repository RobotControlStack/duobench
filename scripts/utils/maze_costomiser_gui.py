import tkinter as tk
import sys
"""this file contains a simple customiser that allows you to use a gui to place horizontal and vetical 
walls on a 10x10 grid representing the mazes board. as of now this script supports the placement of 
walls on the buttom and the left side of a grid cell. """

# ---------------- CONFIG ----------------
GRID_SIZE = 10
CELL_PX = 40

CELL_M = 0.02
ORIGIN_X = -0.09
ORIGIN_Y = 0.09


# ---------------- MUJOCO CONVERSION ----------------
def cell_center(row, col):
    x = ORIGIN_X + (col) * CELL_M
    y = ORIGIN_Y - (row) * CELL_M
    return x, y


def generate_wall_body(row, col, wall_type):
    x, y = cell_center(row, col)

    if wall_type == "h":
        euler = "0 0 90"
    else:
        euler = "0 0 0"

    return f"""
    <body pos="{x:.3f} {y:.3f} 0.003" euler="{euler}">
        <!geom type="mesh" mesh="wall_base" class="visual"/>
        <geom type="mesh" mesh="real_wall" class="visual" rgba="0 1 0 1"/>
        <geom type="mesh" mesh="real_wall" class="collision" friction="0.05 0.05 0.001"/>
    </body>
    """


def inject_xml(xml_path, output_path, walls):
    with open(xml_path, "r") as f:
        xml = f.read()

    insert = ""
    for r, c, t in walls:
        insert += generate_wall_body(r, c, t)

    xml = xml.replace(
        '<body name="board">',
        '<body name="board">\n' + insert
    )

    with open(output_path, "w") as f:
        f.write(xml)

    print("✔ Exported MuJoCo XML →", output_path)


# ---------------- GUI APP ----------------
class App:
    def __init__(self, root, xml_path):
        self.root = root
        self.xml_path = xml_path

        self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
        self.canvas.pack()

        self.grid_origin = (50, 50)

        self.drag_item = None
        self.drag_type = None
        self.offset_x = 0
        self.offset_y = 0

        # final stored walls
        self.walls = []

        # canvas_id → (row, col, type)
        self.wall_items = {}

        self.draw_grid()
        self.create_palette()
        self.create_button()

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    # ---------------- GRID ----------------
    def draw_grid(self):
        gx, gy = self.grid_origin
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x1 = gx + c * CELL_PX
                y1 = gy + r * CELL_PX
                x2 = x1 + CELL_PX
                y2 = y1 + CELL_PX
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="gray")

    # ---------------- PALETTE ----------------
    def create_palette(self):
        self.canvas.create_rectangle(
            600, 100, 640, 108,
            fill="blue", tags=("palette", "h")
        )

        self.canvas.create_rectangle(
            650, 100, 658, 140,
            fill="red", tags=("palette", "v")
        )

    # ---------------- BUTTON ----------------
    def create_button(self):
        tk.Button(self.root, text="Export MuJoCo", command=self.export)\
            .place(x=600, y=500)

    # ---------------- WALL CREATION ----------------
    def create_wall(self, wall_type, x, y):
        if wall_type == "h":
            return self.canvas.create_rectangle(
                x, y, x + CELL_PX, y + 8,
                fill="blue", tags=("wall", "h")
            )
        else:
            return self.canvas.create_rectangle(
                x, y, x + 8, y + CELL_PX,
                fill="red", tags=("wall", "v")
            )

    # ---------------- EVENTS ----------------
    def on_press(self, event):
        item = self.canvas.find_closest(event.x, event.y)[0]
        tags = self.canvas.gettags(item)

        # NEW WALL FROM PALETTE
        if "palette" in tags:
            self.drag_type = "h" if "h" in tags else "v"
            self.drag_item = self.create_wall(self.drag_type, event.x, event.y)

        # EXISTING WALL → REMOVE FROM GRID DATA
        elif "wall" in tags:
            self.drag_item = item
            self.drag_type = "h" if "h" in tags else "v"

            if item in self.wall_items:
                old = self.wall_items[item]
                if old in self.walls:
                    self.walls.remove(old)
                del self.wall_items[item]

        else:
            return

        x1, y1, _, _ = self.canvas.coords(self.drag_item)
        self.offset_x = event.x - x1
        self.offset_y = event.y - y1

    def on_drag(self, event):
        if not self.drag_item:
            return

        x = event.x - self.offset_x
        y = event.y - self.offset_y

        x1, y1, _, _ = self.canvas.coords(self.drag_item)
        self.canvas.move(self.drag_item, x - x1, y - y1)

    def on_release(self, event):
        if not self.drag_item:
            return

        gx, gy = self.grid_origin

        col = (event.x - gx) // CELL_PX
        row = (event.y - gy) // CELL_PX

        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:

            x = gx + col * CELL_PX
            y = gy + row * CELL_PX

            if self.drag_type == "h":
                self.canvas.coords(self.drag_item, x, y + 32, x + CELL_PX, y + CELL_PX)
            else:
                self.canvas.coords(self.drag_item, x, y, x + 8, y + CELL_PX)

            entry = (int(row), int(col), self.drag_type)

            self.walls.append(entry)
            self.wall_items[self.drag_item] = entry

        else:
            self.canvas.delete(self.drag_item)

        self.drag_item = None
        self.drag_type = None

    # ---------------- EXPORT ----------------
    def export(self):
        inject_xml(self.xml_path, "output.xml", self.walls)


# ---------------- MAIN ----------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python maze_costomiser_gui.py <input.xml>")
        sys.exit(1)

    root = tk.Tk()
    root.title("MuJoCo Maze Editor")

    app = App(root, sys.argv[1])

    root.mainloop()