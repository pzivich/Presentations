import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# IMAGE: flow diagram for GAN data simulation
fig, ax = plt.subplots()

# Random noise section
for x in np.arange(0.01, 0.08, 0.01):
    for y in np.arange(0.7, 0.81, 0.01):
        shade = np.random.uniform(0, 1, 1)
        ax.fill_between([x, x+0.01], [y, y], [y+0.01, y+0.01],
                        color='k', alpha=shade[0])

# Generator
ax.arrow(0.1, 0.75, 0.05, 0, head_width=0.01, color='k')
ax.fill_between([0.18, 0.42], [0.8, 0.8], [0.7, 0.7], color='aqua', alpha=0.2)
ax.text(0.20, 0.73, "Generator", size=16)

# Output generator
ax.arrow(0.43, 0.75, 0.05, 0, head_width=0.01, color='k')
ax.text(0.52, 0.66, r"$X^*_1$" + "\n" + r"$X^*_2$" + "\n" + r"$X^*_3$", size=14)

# Real Data
ax.text(0.52, 0.18, r"$X_1$" + "\n" + r"$X_2$" + "\n" + r"$X_3$", size=14)

# Discriminator
ax.arrow(0.58, 0.72, 0.10, -0.15, head_width=0.01, color='k')
ax.arrow(0.58, 0.27, 0.10, 0.15, head_width=0.01, color='k')
ax.fill_between([0.55, 0.83], [0.55, 0.55], [0.45, 0.45], color='orange', alpha=0.2)
ax.text(0.57, 0.48, "Discriminator", size=16)
ax.arrow(0.85, 0.5, 0.05, 0, head_width=0.01, color='k')
ax.text(0.93, 0.485, "T/F", size=14)

ax.set_ylim([-0., 1.])
ax.set_xlim([-0., 1.])
plt.axis('off')
plt.tight_layout()
plt.savefig("images/gan_flow.png", dpi=600, format='png')
plt.close()

# IMAGE: flow diagram for RNN text generation
fig, ax = plt.subplots()

# Step 1: PubMed Query
ax.text(0.02, 0.95, "1: Query PubMed")
rectangle = Rectangle((0., 0.62), 0.95, 0.38, alpha=0.2, color='aqua')
ax.add_patch(rectangle)
ax.text(0.11, 0.86, "1a: Conduct search & extract PubMed IDs")
rectangle = Rectangle((0.1, 0.92), 0.8, -0.08, alpha=0.2, color='blue')
ax.add_patch(rectangle)
ax.text(0.11, 0.76, "1b: Select random sample")
rectangle = Rectangle((0.1, 0.82), 0.8, -0.08, alpha=0.2, color='blue')
ax.add_patch(rectangle)
ax.text(0.11, 0.66, "1c: Pull meta-data from PubMed")
rectangle = Rectangle((0.1, 0.72), 0.8, -0.08, alpha=0.2, color='blue')
ax.add_patch(rectangle)

# Step 2: text processing
ax.text(0.02, 0.55, "2: Text processing")
rectangle = Rectangle((0., 0.32), 0.95, 0.28, alpha=0.2, color='orange')
ax.add_patch(rectangle)
ax.text(0.11, 0.47, "2a: Extract abstracts")
rectangle = Rectangle((0.1, 0.53), 0.8, -0.08, alpha=0.2, color='red')
ax.add_patch(rectangle)
ax.text(0.11, 0.37, "2b: Format text to training data")
rectangle = Rectangle((0.1, 0.43), 0.8, -0.08, alpha=0.2, color='red')
ax.add_patch(rectangle)

# Step 3: train network
ax.text(0.02, 0.24, "3: RNN")
rectangle = Rectangle((0., -0.4), 0.95, 0.7, alpha=0.2, color='green', zorder=1)
ax.text(0.18, -0.1, "Input: \nsent")
ax.text(0.775, -0.1, "Output: \nsentence")

ax.add_patch(rectangle)
ax.text(0.275, 0.14, "sent")
ax.text(0.292, -0.33, "e")
ax.arrow(0.3, 0.12, 0, -0.05, head_width=0.01, color='k')
ax.arrow(0.3, -0.21, 0, -0.05, head_width=0.01, color='k')
ax.scatter([0.30, 0.30, 0.30, 0.30], [-0.15, -0.10, -0.05, 0.],
           marker='o', c='white', edgecolors='k', zorder=2)
rectangle = Rectangle((0.29, -0.2), 0.02, 0.25, alpha=1, facecolor='none', edgecolor='k')
ax.add_patch(rectangle)
ax.arrow(0.315, -0.28, 0.09, 0.4, head_width=0.01, color='k')

ax.text(0.415, 0.14, "ente")
ax.text(0.431, -0.33, "n")
ax.arrow(0.44, 0.12, 0, -0.05, head_width=0.01, color='k')
ax.arrow(0.44, -0.21, 0, -0.05, head_width=0.01, color='k')
ax.scatter([0.44, 0.44, 0.44, 0.44], [-0.15, -0.10, -0.05, 0.],
           marker='o', c='white', edgecolors='k', zorder=2)
rectangle = Rectangle((0.43, -0.2), 0.02, 0.25, alpha=1, facecolor='none', edgecolor='k')
ax.add_patch(rectangle)
ax.arrow(0.45, -0.28, 0.09, 0.4, head_width=0.01, color='k')

ax.text(0.55, 0.14, "nten")
ax.text(0.563, -0.33, "c")
ax.arrow(0.57, 0.12, 0, -0.05, head_width=0.01, color='k')
ax.arrow(0.57, -0.21, 0, -0.05, head_width=0.01, color='k')
ax.scatter([0.57, 0.57, 0.57, 0.57], [-0.15, -0.10, -0.05, 0.],
           marker='o', c='white', edgecolors='k', zorder=2)
rectangle = Rectangle((0.5595, -0.2), 0.02, 0.25, alpha=1, facecolor='none', edgecolor='k')
ax.add_patch(rectangle)
ax.arrow(0.58, -0.28, 0.09, 0.4, head_width=0.01, color='k')

ax.text(0.675, 0.14, "tenc")
ax.text(0.687, -0.33, "e")
ax.arrow(0.695, 0.12, 0, -0.05, head_width=0.01, color='k')
ax.arrow(0.695, -0.21, 0, -0.05, head_width=0.01, color='k')
ax.scatter([0.695, 0.695, 0.695, 0.695], [-0.15, -0.10, -0.05, 0.],
           marker='o', c='white', edgecolors='k', zorder=2)
rectangle = Rectangle((0.6855, -0.2), 0.02, 0.25, alpha=1, facecolor='none', edgecolor='k')
ax.add_patch(rectangle)

ax.set_ylim([-0.6, 1.1])
ax.set_xlim([-0., 0.96])
plt.axis('off')
plt.tight_layout()
plt.savefig("images/rnn_flow.png", dpi=600, format='png')
plt.close()
