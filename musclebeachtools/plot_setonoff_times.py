
import numpy as np
import matplotlib.pyplot as plt
import easygui
import sys
import seaborn as sns

'''PLOT_SETONOFF_TIMES.PY. Function that allows a user to manually label times in which recordings should be considered "on" and "off" in a GUI based format. This program assumes that the user is loading a .npy file with multiple "neuron" class objects that should have fields for spike times, spike amplitudes, and cell quality. The user can either manually enter the file name and location or a GUI file selector will automatically pop up. 

Three modes of on/off setting can be used. 
    I: Selected times will be applied to all cells 
    II: Selected times will be applied to all cells of a given quality (1, 2, or 3) will
    III: Selected times will be applied to a single cell (user enters cell ID) 
    
    NB: this is all very slow because it's built in matplotlib, but due to the necessity of matplotlib on everyone's machines, this should run without much trouble on almost any system within the lab. '''


plt.ion()
global nidx, n, subp, subp1, tag#, cid1, cid2
nidx = 0 # This will index the selected neurons from tcells
tag = np.NaN

datfile = ''

if datfile == '':
    datfile = easygui.fileopenbox()
else:
    pass

print('Loading data.')
n = np.load( datfile, allow_pickle=True)

sp_t = []

def onclick(event):
    # capture the x coordinate (time) of the user click input. This will only matter if the user has already made a keyboard press to indicate on or off time (1 or 0, respectively).
    global ix, iy, tag
    ix, iy = event.xdata, event.ydata
    print("ix ", ix, " iy ", iy)

    ylims = subp.get_ylim()
    ylims1 = subp1.get_ylim()

    if tag == 'on':
        subp.vlines(ix, ylims[0], ylims[1], colors = 'xkcd:seafoam' )
        subp1.vlines(ix, ylims1[0], ylims1[1], colors = 'xkcd:seafoam' )
        #plt.pause(0.01)
        sp_t.append([ix,1])
        tag = np.NaN
        tag_event()

    elif tag == 'off':
        subp.vlines(ix, ylims[0], ylims[1], colors = 'xkcd:vermillion' )
        subp1.vlines(ix, ylims1[0], ylims1[1], colors = 'xkcd:vermillion' )
        #plt.pause(0.01)
        sp_t.append([ix,0])
        tag = np.NaN
        tag_event()

def press(event):
    # Respond to the user's keyboard input. User can select right or left keys to advance/retreat through the individual neurons that meet the initial filtering criteria. User can also press 1 to indicate ON time and 0 to indicate OFF time. Pressing 0 or 1 will then make the code "listen" for a mouse input that is used to log the on/off timestamp. Alternatively, the user can press "z" to delete the most recent time input (mouse click). 
    sys.stdout.flush()

    global nidx, subp, ky, n, tag#, cid1, cid2
    ky = event.key

    if ky=='right':
        nidx+=1
        print(f'Moving to neuron {nidx}.')
        subp.cla()
        subp1.cla()
        plotcell(n[tcells[nidx]])
        
    elif ky=='left':
        nidx-=1
        print(f'Moving to neuron {nidx}.')
        subp.cla()
        subp1.cla()
        plotcell(n[tcells[nidx]])

    elif ky =='0':
        tag = 'off'
        tag_event()
        # set off time

    elif ky=='1':
        tag = 'on'
        tag_event()
    #     # set on time
        
    elif ky=='z':
        tag = 'del'
        tag_event()

    elif ky=='d':
    # This button will save the indicated on/off times to the applicable neurons
        savefunc()


def savefunc():
    # write the user provided on/off times to all of the relevant neurons and save the data at the source. This should only be called if the user has selected 'Y' or 'y' after pressing 'q' to quit the program.   
    finaloots = np.stack(sp_t, axis=0)
    finalons = np.squeeze(np.where(finaloots[:,1]==1))
    finaloffs = np.squeeze(np.where(finaloots[:,1]==0))
    finalons = finalons.tolist()
    finaloffs = finaloffs.tolist()   

    for nrn in tcells:
        n[nrn].on_times.append(finalons)
        n[nrn].off_times.append(finaloffs)
    
    np.save(datfile, n)
    plt.close('all')
    print(f'Data are saved to {datfile}')

def tag_event():
    # Change the sup title at the top of the window to reflect the user's recent selection. This is mostly to show the user that the code has registered their input, however, if the user has selected "z" (delete most recent time stamp), this will clear the last entry in the time stamp list and call the plotting code to refresh the data w/ timestamp deleted.
    global tag

    if tag=='off':
        top_title.update({'text':'Click OFF time.'})
        plt.pause(0.01)
    elif tag=='on':
        top_title.update({'text':'Click ON time.'})
        plt.pause(0.01)
    elif tag=='del':
        del sp_t[-1]
        subp.cla()
        subp1.cla()
        top_title.update({'text':'Deleted last time selection.'})
        plotcell(n[tcells[nidx]])
    elif np.isnan(tag):
        top_title.update({'text':'Ready to continue.'})
        plt.pause(0.01)
        
def plotcell(neuron):
    # Core plotting code to show two subplots. Top subplot is amplitude versus time, and the bottom is firing rate versus time. If there are on/off times, this will also process those and display them accordingly. 

    # Amplitudes subplot:
    meanamp = np.mean(neuron.spike_amplitude)
    stdevamp = np.std(neuron.spike_amplitude)
    
    subp.scatter(neuron.spike_time_sec/3600, neuron.spike_amplitude, color = (0.5,0.5,0.5), marker = '.', alpha = 0.075)
    # set reasonable x and y lims for display. y min is 3rd percentile and max is 5 std
    subp.set_ylim(np.percentile(neuron.spike_amplitude, 3), meanamp+4*stdevamp)
    subp.set_xlim(np.floor(neuron.spike_time_sec[0]/3600), neuron.spike_time_sec[-1]/3600 )

    xlims = subp.get_xlim()
    ylims = subp.get_ylim()

    txtx = xlims[0]+0.1*(xlims[1] - xlims[0])
    txty = ylims[0]+0.5*(ylims[1] - ylims[0])
    subp.text(txtx,txty, f'cluster index: {neuron.clust_idx}')
    subp.set_xlabel('Time (hours)')
    subp.set_ylabel('Amplitude (uV)')
    sns.despine()
    #plt.draw()
    
    # Firing rate subplot:
    t0 = neuron.spike_time_sec[0] / 3600
    t1 = neuron.spike_time_sec[-1] / 3600
    step = 120
    edges = np.arange(t0,t1, step / 3600)
    fr = np.histogram(neuron.spike_time_sec/3600, edges)
    subp1.plot(fr[1][0:-1],fr[0]/step, color = 'xkcd:periwinkle', linewidth = 3)
    subp1.set_ylim(0, np.ceil( (np.max (fr[0])/step )*1.05) )  # Set the limits so they stop drifting when adding vertical lines. 
    subp1.set_xlabel('Time (hours)')
    subp1.set_ylabel('Firing Rate (Hz)')
    sns.despine()

    if np.size(sp_t)>0:
        # Add on/off lines if they exist
        # take the list of on/off times and convert to an array for searching.
        oots = np.stack(sp_t, axis=0)
        tempons = np.squeeze(np.where(oots[:,1]==1))
        tempoffs = np.squeeze(np.where(oots[:,1]==0))
        # pull ylims for the FR plot
        ylims1 = subp1.get_ylim()
        # add the lines to the amplitude plot
        subp.vlines(oots[tempons,0], ylims[0], ylims[1], colors = 'xkcd:seafoam' )
        subp1.vlines(oots[tempons,0], ylims1[0], ylims1[1], colors = 'xkcd:seafoam' )
        # and add the same x vals to the FR plot with the proper ylims
        subp.vlines(oots[tempoffs,0], ylims[0], ylims[1], colors = 'xkcd:vermillion' )
        subp1.vlines(oots[tempoffs,0], ylims1[0], ylims1[1], colors = 'xkcd:vermillion' )
    else:
        pass

    plt.draw()


# Display options for the user to set up filters for viewing.
x = input('SELECT MODE:\n1 (all cells)\n2 (cells of x quality)\n3 (single cell).\n ')

print('Selecting the subset of neurons that fit your selection.')
# Select the subset of neurons that the user wants to work with. 
if x == '1':
    # Get rid of quality 4, but keep all else:
    tcells = []
    ccount = 0
    for i in n:
        if i.quality <4:
            if np.size(tcells) == 0:
                tcells = ccount
            else:
                tcells = np.append(tcells, ccount)
        ccount+=1

elif x == '2':
    # in this case, the user will view all cells of a specific quality. Choose that quality.
    qual = input('Which quality would you like to examine (1, 2, or 3)?\n')
    qual = int(qual)

    tcells = []
    ccount = 0
    for i in n:
        if i.quality == qual:
            if np.size(tcells) == 0:
                tcells = ccount
            else:
                tcells = np.append(tcells, ccount)
        ccount+=1

elif x == '3':
    # ask the user for the cluster idx and check that the user provides an integer. 
    while True:
        temp = input("What is the neuron's clust idx?")
        try:
            val = int(temp)
            print(f'Clust idx is an integer: {val}')
            break
        except ValueError:
            try:
                val = float(temp)
                print(f'Input is a float: {val}. Try an integer.')
            except ValueError:
                print(f'So silly, silly string: {val} Try an integer.')
            pass
    tcells = val

print(f'Done! There are a total of {np.size(tcells)} neurons that meet your criteria.')

print('\n\n\n\n\nINSTRUCTIONS:\nFirst select an action.\nPress 1 to indicate "on" time.\nPress 0 to indicate "off" time.\nClick (mouse) either FR or amplitude plots to place on/off time marker.\nPress "z" to delete the last selected time.\n\n\nPress (space, enter) to continue.')
# Force the user to confirm that s/he has read the instructions:
while True:
    g = input()
    if g == ' ':
        break
    else:
        print('Do you know what the space bar is?')

# Set up the figure and connect it to click and press utilities.
fig = plt.figure(constrained_layout=True, figsize=(14, 7))
top_title = fig.suptitle('Placeholder', fontsize=14)
fig.canvas.mpl_connect('key_press_event', press)
fig.canvas.mpl_connect('button_press_event', onclick)
gs = fig.add_gridspec(2, 1)
subp = fig.add_subplot(gs[0, 0])
subp1 = fig.add_subplot(gs[1, 0], sharex = subp)

# Call the initial plotting:
plotcell(n[tcells[nidx]])


