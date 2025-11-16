import matplotlib.pyplot as plt
import numpy as np

import soraxas_toolbox as st

# create a sin and cos plot
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y1, label="sin")
ax.plot(x, y2, label="cos")
ax.legend()

# uses timg if available
st.image.display(fig)
# uses term_image
st.image.display(fig, backend="term_image")
