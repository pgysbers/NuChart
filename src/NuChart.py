import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import io
import os
# Finds the absolute path where the file is installed is installed
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def text_autoremove(ax, text, xlim, ylim):
  """
  Function to remove text that falls outside the axis automatically.
  Inputs:
   - ax (matplotlib.axis) : axis for the plot
   - text (matplotlib.text): text object
   - xlim (int): maximum value on the x-axis
   - ylim (int): maximum value on the yaxis
  """
  bbox = text.get_window_extent()
  p1 = ax.transData.inverted().transform((bbox.x0, bbox.y0))
  p2 = ax.transData.inverted().transform((bbox.x1, bbox.y1))
  if p1[0] < 0 or p2[0] > xlim:
    text.remove()
  elif p1[1] < 0 or p2[1] > ylim:
    text.remove()

def create_colormap(cols, N):
  """
  Utility function to create a matplotlib colormap from
  a list of discrete colors.
  Inputs:
   - cols (list or str) : List of colors or name of the color
   - N (int) : Number of colors.
  Returns:
    cmap (LinearSegmentedColormap): Colormap
  """
  if not isinstance(cols, list) and N == 1:
    cols = [cols, cols]
    N = 2
  assert (len(cols) == N)
  cmap = LinearSegmentedColormap.from_list(" ", cols, N=N)
  return cmap


class NuChart():
  """
  Class to facilitate plotting of nuclear charts. Uses data from ame2020 (Chinese Phys. C 45 030002 (2021)) 
  and nubase 2020 (Chinese Physics C 45 (2021) 030001). Also contains results of ab initio methods through the year.
  Plots is based on plt.pcolormesh with use of Pillow to add drop shadow automatically to
  the nuclear chart.
  """

  def __init__(self, xlim=160, ylim=100, figsize=(12,8), shadow=True, shadowtone=150, blur_radius=80, base_legend=True, alpha_stable=0.5):
    """
    Initialiazation of the NuChart class. Creates the base figure with the AME2020 data.
    Inputs:
      - xlim (int): Maximal value for the x-axis (neutrons) that will be plotted.
      - ylim (int): Maximal value for the y-axis (protons) that will be plotted.
      - figsize (tuple(float)): Size of the figure
      - shadow (bool): Drop Shadow will be added to the plot with Pillow if True
      - shadowtone (int): Color for the shadow. Smaller number will be darker
      - blur_radius (float): Radius for the shadow
      - base_legend (bool): Legend for the AME2020 values will be plotted if True.
      - alpha_stable (float): alpha value to highlight the stable isotopes
    """
    self.ame20 = pd.read_csv(ROOT_DIR+'/Data/ame20.csv')
    self.nubase20 = pd.read_csv(ROOT_DIR+'/Data/nubase20.csv')
    self.stable = self.nubase20[(
        self.nubase20.t_1_2 == 'stbl') & (self.nubase20.extrapol == False)]
    self.xlim = xlim
    self.ylim = ylim
    self.shadowtone=shadowtone
    self.blur_radius = blur_radius
    self.base_legend = base_legend
    self.edgewidth = 0.5
    self.alpha = alpha_stable
    self.candidate_markers = []
    self.candidate_colors = []
    self.candidate_labels = []
    self.legend_size = 14
    self._create_base_plot(figsize=figsize, shadow=shadow)


  def _create_base_plot(self, figsize=(12, 8), shadow=True):
    """
    Creates the base figure with the AME2020 data.
    Inputs:
      - figsize (tuple(float)): Size of the figure
      - shadow (bool): Drop Shadow will be added to the plot with Pillow if True
    """
    self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
    self.ax.use_sticky_edges = False
    # Add nuclei from AME2020
    for index, row in self.ame20.iterrows():
      if row['extrapol'] == False:
        if (row['N'] < self.xlim) and (row['Z'] < self.ylim):
          self.ax.add_patch(plt.Rectangle((row["N"]-0.5, row['Z']-0.5), 1, 1, fc='silver',
                                    ec='w', lw=self.edgewidth, clip_on=True))
      else:
        if (row['N'] < self.xlim) and (row['Z'] < self.ylim):
          self.ax.add_patch(plt.Rectangle((row["N"]-0.5, row['Z']-0.5), 1, 1, fc='gainsboro',
                                    ec='w', lw=self.edgewidth, clip_on=True))
    # Set limits
    self.ax.set_xlim(-0.5, self.xlim-0.5)
    self.ax.set_ylim(-0.5, self.ylim-0.5)
    # Add the shadow to the plot
    if shadow:
      #Remove everything to figure except the nuclear chart to which we
      #want to add the shadow
      self.ax.set_frame_on(False)
      xticks = self.ax.get_xticks()
      yticks = self.ax.get_yticks()
      self.ax.get_xaxis().set_ticks([])
      self.ax.get_yaxis().set_ticks([])
      #Add the shadow
      img = self.add_shadow()
      #Add back the nuclear chart for the pillow image
      self.ax.clear()
      self.ax.imshow(img, extent=(-0.5, self.xlim-0.5, -0.5, self.ylim-0.5))
      #Add the frames and ticks back
      self.ax.set_frame_on(True)
      self.ax.set_xticks(xticks)
      self.ax.set_yticks(yticks)
      self.ax.set_xlim(-0.5, self.xlim-0.5)
      self.ax.set_ylim(-0.5, self.ylim-0.5)

    # Add stable nuclei
      X = np.arange(0,self.stable['N'].max()+1,1)
      Y = np.arange(0,self.stable['Z'].max()+1,1)
      Z = np.full([self.stable['Z'].max()+1, self.stable['N'].max()+1], np.nan)
      coord = self.stable[['Z', 'N']].to_numpy()
      for i in coord:
        Z[i[0],i[1]] = 1
      cmap = create_colormap(['gray', 'gray'], 2)
      bounds = np.arange(2)
      norm = colors.BoundaryNorm(boundaries=bounds, ncolors=2)
      c = self.ax.pcolor(X, Y, Z, shading='nearest', norm=norm,
                    cmap=cmap, edgecolors='w', linewidth=self.edgewidth, alpha=self.alpha, zorder=20)

    # Set labels
    self.ax.set_xlabel('N', fontsize=16)
    self.ax.set_ylabel('Z', fontsize=16)
    # Set ticks
    self.ax.tick_params(axis='both', which='major', labelsize=12)
    if self.base_legend:
      self.add_base_legend()


  def get_axis(self):
    # Return the axis of the plot if needed.
    return self.ax
  

  def get_fig(self):
    # Return the figure of the plot if needed.
    return self.fig
  

  def add_shadow(self):
    """
    Adds the drop shadow to the nuclear chart plot. First save the figure as a PNG to
    a buffer and load it as an image in Pillow. Then add gaussian blur in Pillow and return
    the image.
    """
    canvas = self.fig.canvas
    buffer = io.BytesIO()
    # Save as PNG to the buffer
    self.fig.savefig(buffer, format='png', transparent=True,
                bbox_inches='tight', dpi=500, pad_inches=0)
    buffer.seek(0)  # Rewind the buffer to the beginning
    img = Image.open(buffer)
    # Create array from image and set it to gray scale to tone we want
    blurred_img = np.array(img)
    blurred_img[:, :, :3] = self.shadowtone
    # Add gaussian blur
    blurred_img = Image.fromarray(np.array(blurred_img))
    blurred_img = blurred_img.filter(
        ImageFilter.GaussianBlur(radius=self.blur_radius))
    # Mix the shadow with original plot
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(blurred_img, mask=blurred_img.split()[3])
    background.paste(img, mask=img.split()[3])
    buffer.close()
    return background
  

  def add_base_legend(self):
    "Add the legend for the AME2020 points."
    legend_elements = [Line2D([0], [0], marker='s', color='w', label='Stable nuclei',
                              markerfacecolor='gray', markersize=15, alpha=0.8, linestyle='none'),
                      Line2D([0], [0], marker='s', color='w', label='AME 2020',
                              markerfacecolor='silver', markersize=15, linestyle='none'),
                      Line2D([0], [0], marker='s', color='w', label='AME 2020 (extrapolated)',
                              markerfacecolor='gainsboro', markersize=15, linestyle='none')]
    base_legend = plt.legend(handles=legend_elements, loc='upper left',
              fontsize=self.legend_size, frameon=False)
    self.ax.add_artist(base_legend)


  def add_magic_numbers(self):
    "Add lines at magic numbers. Text is also added if spacing allows it."
    proton_magic = [2, 8, 20, 28, 40, 50, 82]
    proton_line_min = [0, 0, 7, 11, 29, 45, 91]
    proton_line_max = [12, 24, 54, 74, 100, 100, 150]
    neutron_magic = [2, 8, 20, 28, 40, 50, 82, 126, 184]
    neutron_line_min = [0, 0, 4, 7, 13, 17, 32, 74, 74]
    neutron_line_max = [12, 24, 32, 38, 48, 54, 80, 100, 100]

    for p, pmin, pmax in zip(proton_magic, proton_line_min, proton_line_max):
      if p < self.ylim:
        self.ax.plot([pmin, pmax], [p, p], color='darkgrey',
                linestyle="--", lw=2, alpha=0.9, zorder=30)
        text = self.ax.text(pmin-1, p-0.5, f'Z={p}', color='k', fontsize=10, ha="right")
        text_autoremove(self.ax, text, self.xlim, self.ylim)
    for n, nmin, nmax in zip(neutron_magic, neutron_line_min, neutron_line_max):
      if n < self.xlim:
        self.ax.plot([n, n], [nmin, nmax],  color='darkgrey',
                linestyle="--", lw=2, alpha=0.9, zorder=30)
        # if n >= minnlabel:
        text = self.ax.text(n-4, nmin-2, f'N={n}', color='k', fontsize=10)
        # !!! get the extent of the text
        text_autoremove(self.ax, text, self.xlim, self.ylim)


  def plot_abinitio(self, cmap="viridis", years=[2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024, 2025], year_max=None):
    """
    Plots the reach of ab initio methods through the years. 
    Inputs:
      - cmap (matplotlib.colormap): Colormap for the plot.
      - years (list): Years to have included on the plot.
    """
    df = pd.read_csv(ROOT_DIR+'/Data/abinitiosummary.csv', on_bad_lines='warn')
    df['n'] = df['a'] - df['z']
    xlim = max(self.xlim, df['n'].max()+1)
    ylim = max(self.ylim, df['z'].max()+1)
    X = np.arange(0, xlim, 1)
    Y = np.arange(0, ylim, 1)
    Z = np.full([ylim, xlim], 0,dtype=float)

    for j, y in enumerate(years[::-1]):
      df_year = df[df['year'] <= y].copy()
      coord_year = df_year[['z', 'n']].to_numpy()
      for i in coord_year:
        Z[i[0], i[1]] = j+0.5
        if year_max and y > year_max:
          Z[i[0], i[1]] = 0
    Z[Z == 0] = np.nan

    cmap = plt.get_cmap(cmap, len(years))
    bounds = np.arange(len(years)+1)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(years))


    c = self.ax.pcolor(X, Y, Z, shading='nearest', norm=norm,
                  cmap=cmap, edgecolors='w', linewidth=self.edgewidth)
    self.add_legend_colormap(cmap, years[::-1])


  def plot_ncsm(self, cmap='nipy_spectral', Nmax=0):
    df = pd.read_csv(ROOT_DIR+'/Data/ncsm_eMax08.csv', on_bad_lines='warn')
    xlim = max(self.xlim, df['N'].max()+1)
    ylim = max(self.ylim, df['Z'].max()+1)

    X = np.arange(0, xlim, 1)
    Y = np.arange(0, ylim, 1)
    Z = np.full([ylim, xlim], 0, dtype=float)

    df_Nmax = df[df['Nmax'] == Nmax].dropna().copy()
    coord_Nmax = df_Nmax[['Z', 'N']].to_numpy()

    for i, row in df_Nmax.iterrows():
      print(i,row)
      z = int(row['Z'])
      n = int(row['N'])
      Z[n,z] = row['dim']

    Z[Z<=0] = np.nan

    max_magnitude = 10
    cmap = plt.get_cmap(cmap, max_magnitude)
    bounds = np.arange(max_magnitude+1)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=max_magnitude)

    c = self.ax.pcolor(X, Y, np.log10(Z), shading='nearest', norm=norm,
                       cmap=cmap, edgecolors='w', linewidth=self.edgewidth)
    self.add_legend_colormap(cmap, [f'$10^{s}$' for s in range(max_magnitude)])
    self.ax.set_title(f'No. of SDs for Nmax={Nmax}')


  def add_legend_colormap(self, cmap, labels):
    try:
      legend_elements = [Line2D([0], [0], marker='s', color='w', label=label,
                                markerfacecolor=cmap.colors[i], markersize=15, alpha=0.8, linestyle='none') for i, label in enumerate(labels)]
    except AttributeError:
      legend_elements = [Line2D([0], [0], marker='s', color='w', label=label,
                                markerfacecolor=cmap(np.linspace(0, 1, cmap.N))[i], markersize=15, alpha=0.8, linestyle='none') for i, label in enumerate(labels)]
    legend = plt.legend(handles=legend_elements, loc='lower right',
                        fontsize=self.legend_size, frameon=False)
    self.ax.add_artist(legend)

  def add_candidate_legend(self):
    legend_elements = [Line2D([0], [0], marker=marker, label=label,
                              color=color, markersize=15, linestyle='none') 
                              for marker, label, color in 
                              zip(self.candidate_markers, self.candidate_labels,self.candidate_colors)]
    legend = plt.legend(handles=legend_elements, loc='lower center',
                        fontsize=self.legend_size, frameon=False)
    self.ax.add_artist(legend)


  def plot_Z(self, X, Y, Z, cols, labels=None):
    if isinstance(cols, str):
      N = 2
      cols = [cols, cols]
    else:
      N = len(cols)
    cmap = create_colormap(cols, N)
    bounds = np.arange(N+1)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=N)
    c = self.ax.pcolor(X, Y, Z, shading='nearest', norm=norm,
                  cmap=cmap, edgecolors='w', linewidth=0.5)
    if labels:
      self.add_legend_colormap(cmap, labels)

  
  def plot_observable(self, df, color='tab:blue', label=None):
    try:
      assert('n' in df.columns)
    except AssertionError:
      print("'n' must be in the columns of dataframe")
    try:
      assert ('z' in df.columns)
    except AssertionError:
      print("'z' must be in the columns of dataframe")
    Zmax = df['z'].max()+1
    Nmax = df['n'].max()+1
    X = np.arange(0, Nmax, 1)
    Y = np.arange(0, Zmax, 1)
    coord = df[['z', 'n']].to_numpy()
    Z = np.full([Zmax, Nmax], np.nan)
    for i in coord:
      Z[i[0], i[1]] = 1
    self.plot_Z(X,Y,Z,color,[label])

  def savefig(self, name, dpi=300):
    self.fig.tight_layout()
    plt.savefig(name,
            bbox_inches='tight', dpi=dpi, pad_inches=0)
    
  def highlight_candidates(self, candidates, marker="*", color="r", label = None, legend=True):
    Z_values, A_values = zip(*candidates)
    Z_values = np.array(Z_values)
    A_values = np.array(A_values)
    N_values = A_values-Z_values
    self.candidate_colors.append(color)
    self.candidate_markers.append(marker)
    if label:
      self.candidate_labels.append(label)
    self.ax.scatter(N_values, Z_values, marker=marker, color=color, zorder=30)
    if legend:
      self.add_candidate_legend()
      
    



      
