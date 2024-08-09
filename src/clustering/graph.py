import os
import pandas as pd
import networkx as nx
from kmeans import Clustering
import matplotlib.pyplot as plt

root_path = os.path.join(os.path.dirname(__file__), "../../")
data = pd.read_csv(f"{root_path}/data/processed/dataset_voyage.csv")[:559]
triplet = pd.read_csv(f"{root_path}/data/processed/triplet.csv")[:722]
capec_category = pd.read_csv(f"{root_path}/data/processed/capec_category_voyage.csv")
capec_domain_mechanism = pd.read_csv(f"{root_path}/data/raw/capec_domain_mechanism.csv")

cluster = Clustering(
            data=data,
            categoty=capec_category,
            domain_mechanism=capec_domain_mechanism
        )

colors = [
    '#FF0000',  # Red
    '#00FF00',  # Lime
    '#0000FF',  # Blue
    '#FFFF00',  # Yellow
    '#FF00FF',  # Magenta
    '#00FFFF',  # Cyan
    '#800000',  # Maroon
    '#008000',  # Green
    '#000080',  # Navy
    '#808000',  # Olive
    '#800080',  # Purple
    '#008080',  # Teal
    '#FFA500',  # Orange
    '#A52A2A',  # Brown
    '#FFC0CB',  # Pink
    '#E6E6FA',  # Lavender
    '#FFD700',  # Gold
    '#C0C0C0',  # Silver
    '#228B22',  # Forest Green
    '#4B0082',  # Indigo
    '#FF4500',  # Orange Red
    '#2E8B57',  # Sea Green
    '#8A2BE2',  # Blue Violet
    '#20B2AA',  # Light Sea Green
    '#FF69B4',  # Hot Pink
    '#00CED1',  # Dark Turquoise
    '#FF1493',  # Deep Pink
    '#00FA9A',  # Medium Spring Green
    '#1E90FF',  # Dodger Blue
    '#B22222',  # Fire Brick
]

class Network:
    def __init__(self) -> None:
        self.x, self.cluster_index = cluster.run()
        self.G = nx.Graph()
        
        # self.cluster_index = {1: [1, 2, 21, 24, 25, 27, 29, 30, 41, 42, 56, 137, 150, 154, 181, 237, 253, 264, 270, 280, 286, 297, 347, 355, 367, 423, 469, 500, 531, 532, 536, 538, 539, 540, 541, 550, 552], 2: [3, 9, 10, 12, 36, 37, 38, 39, 40, 53, 76, 77, 80, 82, 85, 86, 93, 97, 102, 103, 114, 129, 131, 138, 139, 140, 141, 142, 143, 144, 146, 147, 148, 149, 200, 238, 244, 287, 295, 298, 330, 369, 376, 378, 379, 380, 383, 384, 390, 403, 411, 413, 417, 418, 420, 421, 422, 426, 438, 453, 454, 458, 470, 489, 490, 507, 524, 526, 543, 544, 546, 548], 3: [11, 43, 44, 45, 49, 64, 68, 69, 71, 78, 87, 88, 89, 90, 109, 111, 145, 151, 155, 156, 157, 158, 170, 216, 240, 241, 268, 269, 271, 272, 273, 274, 275, 276, 277, 278, 279, 281, 282, 283, 284, 306, 310, 340, 342, 343, 344, 345, 346, 348, 349, 350, 351, 352, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 374, 375, 385, 395, 396, 402, 407, 408, 409, 410, 414, 416, 429, 430, 431, 432, 434, 442, 464, 465, 467, 479, 486, 488, 493, 499, 501, 502, 503, 504, 505, 508, 509, 522, 528, 533, 534, 535, 542, 553, 556], 4: [7, 13, 15, 20, 22, 26, 32, 33, 35, 50, 63, 81, 101, 104, 116, 118, 119, 123, 124, 128, 130, 132, 133, 159, 160, 161, 163, 164, 165, 166, 211, 215, 217, 291, 292, 296, 308, 311, 313, 314, 315, 316, 318, 319, 320, 321, 322, 323, 324, 328, 329, 353, 381, 433, 449, 491, 495, 496, 514], 5: [14, 62, 105, 107, 117, 167, 193, 220, 243, 246, 247, 248, 249, 317, 331, 377, 391, 527, 551], 6: [0, 4, 6, 8, 16, 17, 23, 31, 65, 73, 79, 83, 100, 106, 108, 112, 113, 115, 121, 122, 127, 134, 135, 136, 153, 182, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 239, 288, 293, 294, 309, 333, 337, 339, 341, 382, 386, 387, 388, 389, 404, 415, 419, 427, 428, 437, 448, 455, 472, 474, 475, 481, 483, 484, 492, 494, 498, 510, 511, 512, 513, 515, 529, 530, 537, 547, 554], 7: [5, 34, 48, 51, 55, 57, 61, 66, 67, 75, 84, 98, 99, 120, 125, 162, 219, 250, 251, 252, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 265, 266, 267, 285, 299, 301, 302, 303, 304, 305, 307, 312, 332, 334, 335, 336, 370, 371, 412, 424, 425, 439, 443, 444, 445, 456, 457, 459, 460, 461, 462, 471, 485, 487, 497, 516, 517, 518, 519, 521, 523, 549, 558], 8: [18, 19, 28, 46, 47, 52, 54, 58, 59, 60, 70, 72, 74, 91, 92, 94, 95, 96, 110, 126, 152, 168, 169, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 194, 195, 196, 197, 198, 199, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 212, 213, 214, 218, 242, 245, 289, 290, 300, 325, 326, 327, 338, 354, 366, 368, 372, 373, 392, 393, 394, 397, 398, 399, 400, 401, 405, 406, 435, 436, 440, 441, 446, 447, 450, 451, 452, 463, 466, 468, 473, 476, 477, 478, 480, 482, 506, 520, 525, 545, 555, 557]}
        
        self.dic = {}
        
        for key, nodes in self.cluster_index.items():
            for node in nodes:
                self.dic[node] = key
        
        self.make_graph()
        
    def make_graph(self):
        edges = []
        edges_type = []
        nodes = []
        
        self.node_colors = []
        
        for index, node in enumerate(data["ID"].tolist()):
            nodes.append(node[5:])
            cluster_num = self.dic[index]
            self.node_colors.append(colors[cluster_num-1])
        
        self.G.add_nodes_from(nodes)
        
        data["ID"].tolist()
        
        for series in triplet.to_records():
            if series["Relation"] == "ParentOf":
                edges.append((series["ID1"][5:], series["ID2"][5:]))
        self.G.add_edges_from(edges)
        
        self.pos = {}
        for index, series in enumerate(data.to_records()):
            self.pos[series["ID"][5:]] = self.x[index]
        
        
    def draw(self):
        plt.figure(figsize=(70, 70))
        nx.draw_networkx(self.G, pos=self.pos, node_color=self.node_colors, node_size=1200)
        plt.savefig("./graph.png")
        
    def analysis(self):
        deg_center = nx.betweenness_centrality(self.G)
        deg_center_sorted = sorted(deg_center.items(), key=lambda x:x[1], reverse=True)
        print(deg_center_sorted[:20])

        deg_center = nx.degree_centrality(self.G)
        deg_center_sorted = sorted(deg_center.items(), key=lambda x:x[1], reverse=True)
        print(deg_center_sorted[:20])


Network().draw()