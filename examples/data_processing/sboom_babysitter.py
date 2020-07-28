import os, glob
import time
import math
import pickle
import win32api
import subprocess as sp
# from wing_model import model

def enhanced_waitForCompletion(processes_to_track = ['xfoil.exe', 'python.exe'],
                                processes_to_kill = ['xfoil.exe', 'python.exe'],
                                increment_time=30., max_time=3600., not_on_kill_list = []):
    flag = False
    # Need to wait to get into standard

    current_time = 0
    process_exists = True
    print('Start waiting')
    while max_time>current_time and process_exists == True:
        time.sleep(increment_time)
        current_time += increment_time
        tasklistrl = os.popen("tasklist").readlines()

        process_exists = False
        for process in processes_to_track :
            for examine in tasklistrl:
                if process == examine[0:len(process)]:
                    pid = int(examine[29:34])
                    if pid not in not_on_kill_list:
                        process_exists = True
        if process_exists:
            print('Current run time: ' + str(current_time))
        else:
            print('Current run time: ' + str(current_time) + '. DONE!')

        
    # If max running time was achieved and still not done, KILL!!
    if max_time>=current_time and process_exists == True:
        
        flag = True
        print ('TIME TO KILL')
        for process in processes_to_kill:
            try:
                kill(str(process), tasklistrl, not_on_kill_list)
            except:
                pass
         
    # If done, why even complain?! Just get out of there
    
    return flag

def kill(process, tasklistrl=None, not_on_kill_list = []):
    """This function has a dependency of win32api (which abaqus already has).
       To install, just use pip install pypiwin32"""
    if tasklistrl == None:
        tasklistrl = os.popen("tasklist").readlines()
    process_exists_forsure = False
    gotpid = False
    for examine in tasklistrl:
        if process == examine[0:len(process)]:
            process_exists_forsure = True
    if process_exists_forsure:
        print("That process exists.")
    else:
        print("That process does not exist.")
        # sys.exit()
    for getpid in tasklistrl:
        if process == getpid[0:len(process)]:
            pid = int(getpid[29:34])
            if pid not in not_on_kill_list:
                gotpid = True
                try:
                    handle = win32api.OpenProcess(1, False, pid)
                    win32api.TerminateProcess(handle, 0)
                    win32api.CloseHandle(handle)
                    print("Successfully killed process %s on pid %d." % (getpid[0:len(process)], pid))
                except win32api.error as err:
                    print(err)
                    # sys.exit()
    if not gotpid:
        print("Could not get process pid.")

def get_good_pids(process, tasklistrl=None):
    """This function has a dependency of win32api (which abaqus already has).
       To install, just use pip install pypiwin32"""
    if tasklistrl == None:
        tasklistrl = os.popen("tasklist").readlines()

    process_exists_forsure = False
    gotpid = False
    for examine in tasklistrl:
        if process == examine[0:len(process)]:
            process_exists_forsure = True
    if process_exists_forsure:
        print("That process exists.")
    else:
        print("That process does not exist.")
        # sys.exit()
    good_guys = [] 
    for getpid in tasklistrl:
        if process == getpid[0:len(process)]:
            pid = int(getpid[29:34])
            gotpid = True
            good_guys.append(pid)
    if not gotpid:
        print("Could not get process pid.")
    return good_guys

def run():
    # Define work directory (currently uses the command line one)
    current_dir = os.path.dirname(os.path.realpath('__file__'))
    # Command to execute
    f = open('mode.txt','r')
    mode = f.read()
    f.close()
    if mode=='p':
        main_script = 'decoded_PLdB_pred_babysit.py'
    elif mode=='t':
        main_script = 'decoded_PLdB_true_babysit.py'
    # Directory where the new command line runs
    popen_dir = current_dir
    # # Input file for Abaqus
    # input_file = os.path.join(current_dir, 'input.p')
    # # Output file from Abaqus
    # output_file = os.path.join(current_dir, 'output.p')
    # Command to execute
    command = 'python ./' + main_script
    # Time to wait for termination
    time_terminate = 30.
    # Time increment for checking (if too small, will not work)
    increment_time=2.
    # # delete previous input/output files
    # try:
        # os.remove(input_file)
    # except OSError:
        # pass
    # try:
        # os.remove(output_file)
    # except OSError:
        # pass
    
    # Check for executables that were running before so that
    # you do not kill another one by accident
    not_on_kill_list = get_good_pids('python.exe')
    
    # Run abaqus script
    ps = sp.Popen(command, cwd = popen_dir, shell=True)
    
    # Wait for termination
    terminated = enhanced_waitForCompletion(processes_to_track = ['xfoil.exe', 'python.exe'],
                                            processes_to_kill = ['xfoil.exe', 'python.exe'],
                                            max_time=time_terminate, not_on_kill_list = not_on_kill_list,
                                            increment_time = increment_time)
    # # If job is killed or if it did not converge, dummy outputs are generated
    # if terminated:
        # for filename in glob.glob("Polar_*"):
            # os.remove(filename) 
        # for filename in glob.glob("Cp_*"):
            # os.remove(filename) 
        # f = open('outputs.txt','w')
        # f.write('%6.5f\t%6.5f\t%6.5f\t%6.5f\t%6.5f' % (-999, 1, 0, 0, 1)) # the final output denotes that XFOIL was terminated due to time constraints
        # f.close()       

if __name__ == '__main__':
    # dummy examples. Inputs and outputs are all dictionaries
    
    run()
