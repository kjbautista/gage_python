from __future__ import print_function, division
from builtins import int
try:
    from configparser import ConfigParser # needed to read ini files
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0

import GageSupport as gs
import numpy as np
import threading
import sys
import time
from os import fsync
import platform

from GageConstants import (CS_CURRENT_CONFIGURATION, CS_ACQUISITION_CONFIGURATION,
                           CS_STREAM_TOTALDATA_SIZE_BYTES, CS_DATAPACKING_MODE,
                           CS_GET_DATAFORMAT_INFO, CS_BBOPTIONS_STREAM, CS_MODE_USER1,
                           CS_MODE_USER2, CS_EXTENDED_BOARD_OPTIONS, STM_TRANSFER_ERROR_FIFOFULL)
                            
from GageErrors import (CS_MISC_ERROR, CS_INVALID_PARAMS_ID,
                        CS_STM_TRANSFER_TIMEOUT, CS_STM_COMPLETED)


# Code used to determine if python is version 2.x or 3.x
# and if os is 32 bits or 64 bits.  If you know they
# python version and os you can skip all this and just
# import the appropriate version

os_name = platform.system()

if os_name == 'Windows':
    is_64_bits = sys.maxsize > 2**32

    if is_64_bits:
        if sys.version_info >= (3, 0):
            import PyGage3_64 as PyGage
        else:
            import PyGage2_64 as PyGage
    else:        
        if sys.version_info > (3, 0):
            import PyGage3_32 as PyGage
        else:
            import PyGage2_32 as PyGage
else:
    import PyGage          


TRANSFER_TIMEOUT = 10000 # milliseconds
STREAM_BUFFERSIZE = 0x200000 # 2097152
OUT_FILE = "Data"

g_cardTotalData = []

def save_data_to_file(handle, card_index, sample_size, app, stream_started_event,
                   ready_for_stream_event, stream_aborted_event, stream_error_event):
    # buffer size for this call is in bytes
    buffer1 = PyGage.GetStreamingBuffer(handle, card_index, app['BufferSize'])
    if isinstance(buffer1, int):  # in python 2 would be isinstance(buffer1, (int, long))
        print("Error getting streaming buffer 1: ", PyGage.GetErrorString(buffer1))
        stream_error_event.set()
        time.sleep(1) # to give stream_error_wait() a chance to catch it
        return False
    # buffer size for this call is in bytes        
    buffer2 = PyGage.GetStreamingBuffer(handle, card_index, app['BufferSize'])    
    if isinstance(buffer2, int):
        print("Error getting streaming buffer 2: ", PyGage.GetErrorString(buffer2))
        PyGage.FreeStreamingBuffer(handle, card_index, buffer1)
        stream_error_event.set()
        time.sleep(1) # to give stream_error_wait() a chance to catch it        
        return False

    filename = app['DataFile'] + '.dat'
    if app['SaveToFile'] == 1:
        try:
            f = open(filename, 'wb')
        except IOError:
            print("Error opening ", filename)
            PyGage.FreeStreamingBuffer(handle, card_index, buffer1)
            PyGage.FreeStreamingBuffer(handle, card_index, buffer2)
            stream_error_event.set()
            time.sleep(1) # to give stream_error_wait() a chance to catch it            
            return False

    buffer = np.zeros_like(buffer1)
    work_buffer = np.zeros_like(buffer1)
    
    ready_for_stream_event.set()

    transfer_size = app['BufferSize'] // sample_size
  
    stream_started_event.wait() # should also be waiting for abort
    done = False
    stream_completed = False
    loop_count = 0
    total_data = 0
    work_buffer_active = False
    while not done and not stream_completed:
        set = stream_aborted_event.wait(0)
        if set: # user has aborted
            done = True
            break
        # on any error, set an error event
        if loop_count & 1:
            buffer = buffer2
        else:
            buffer = buffer1

        status = PyGage.TransferStreamingData(handle, card_index, buffer, transfer_size)

        if status < 0:
            if status == CS_STM_COMPLETED:
                stream_completed = True
            else:
                print("Error: ", PyGage.GetErrorString(status))
                stream_error_event.set()
                time.sleep(1) # to give stream_error_wait() a chance to catch it                
                break		

        if app['SaveToFile'] != 0 and work_buffer_active:
            try:
                work_buffer.tofile(f)
                if app['FileFlagNoBuffering'] != 0:
                    f.flush()
                    fsync(f)
            except IOError:
                stream_error_event.set()
                time.sleep(1) # to give stream_error_wait() a chance to catch it                
                done = True

            
        p = PyGage.GetStreamingTransferStatus(handle, card_index, app['TimeoutOnTransfer'])
        if isinstance(p, tuple):  
            g_cardTotalData[card_index-1] += p[1]  # have total_data be an array, 1 for each card
            if p[2] == 0:
                stream_completed = False
            else:
                stream_completed = True

            if STM_TRANSFER_ERROR_FIFOFULL & p[0]:
                if app['ErrorHandlingMode'] != 0:
                    stream_error_event.set()
                    print("Fifo full detected on card ", card_index)
                    done = True
                    time.sleep(1) # to give stream_error_wait() a chance to catch it

        else: # error detected
            stream_error_event.set()
            done = True
            if p == CS_STM_TRANSFER_TIMEOUT:
                print("\nStream transfer timeout on card ", card_index)
            else:
                set = stream_aborted_event.wait(0)
                if not set: # user has aborted
                    print("Error: ", PyGage.GetErrorString(p))
            time.sleep(1) # to give stream_error_wait() a chance to catch it

        work_buffer = buffer
        work_buffer_active = True
        loop_count = loop_count + 1      

    # write out work buffer
    if stream_completed and app['SaveToFile'] != 0:
        try: # the amount we save here might be less than the work_buffer size
            write_size = p[1] * sample_size
            save_buffer = work_buffer[:write_size]
            save_buffer.tofile(f)
            if app['FileFlagNoBuffering'] != 0:
                f.flush()
                fsync(f)
        except IOError:
            print("\nError writing file on card ", card_inex)
            stream_error_event.set()
            time.sleep(1) # to give stream_error_wait() a chance to catch it

    if app['SaveToFile'] != 0:
        f.close()

    status = PyGage.FreeStreamingBuffer(handle, card_index, buffer1)
    status = PyGage.FreeStreamingBuffer(handle, card_index, buffer2)
    if stream_completed:
        return True
    else:
        return False
    

def update_progress(elapsedTime, totalBytes):
    hours = 0
    minutes = 0
    seconds = 0

    if elapsedTime > 0:
        rate = (totalBytes / 1000000) / (elapsedTime)
        seconds = int(elapsedTime) # elapsed time is in seconds
        if seconds >= 60:  # seconds
            minutes = seconds // 60
            if minutes >= 60:
                hours = minutes // 60
                if hours > 0:
                    minutes %= 60
            seconds %= 60
        total = totalBytes / 1000000  # mega samples
        s = "Total: {0:.2f} MB, Rate: {1:6.2f} MB/s Elapsed time: {2:d}:{3:02d}:{4:02d}\r".format(total,
                                                                                                  rate,
                                                                                                  hours,
                                                                                                  minutes,
                                                                                                  seconds)
        sys.stdout.write(s)
        sys.stdout.flush()

   
def configure_system(handle, filename):
    acq, sts = gs.LoadAcquisitionConfiguration(handle, filename)	
    if isinstance(acq, dict) and acq:	
        status = PyGage.SetAcquisitionConfig(handle, acq)
        if status < 0:
            return status
    else:
        print("Using defaults for acquisition parameters")
        status = PyGage.SetAcquisitionConfig(handle, acq)

    if sts == gs.INI_FILE_MISSING:
        print("Missing ini file, using defaults")
    elif sts == gs.PARAMETERS_MISSING:
        print("One or more acquisition parameters missing, using defaults for missing values")                
 
    system_info = PyGage.GetSystemInfo(handle)

    if not isinstance(system_info, dict): # if it's not a dict, it's an int indicating an error
        return system_info

    channel_increment = gs.CalculateChannelIndexIncrement(acq['Mode'],
                                                          system_info['ChannelCount'],
                                                          system_info['BoardCount'])
	
    missing_parameters = False
    for i in range(1, system_info['ChannelCount'] + 1, channel_increment):
        chan, sts = gs.LoadChannelConfiguration(handle, i, filename)
        if isinstance(chan, dict) and chan:
            status = PyGage.SetChannelConfig(handle, i, chan)
            if status < 0:
                return status
        else:
            print("Using default parameters for channel ", i)

        if sts == gs.PARAMETERS_MISSING:
            missing_parameters = True

    if missing_parameters:
        print("One or more channel parameters missing, using defaults for missing values")
	
    
    missing_parameters = False
    # in this example we're only using 1 trigger source, if we use 
    # system_info['TriggerMachineCount'] we'll get warnings about 
    # using default values for the trigger engines that aren't in
    # the ini file
    trigger_count = 1    
    for i in range(1, trigger_count + 1):
        trig, sts = gs.LoadTriggerConfiguration(handle, i, filename)
        if isinstance(trig, dict) and trig:
            status = PyGage.SetTriggerConfig(handle, i, trig)
            if status < 0:
                return status
        else:
            print("Using default parameters for trigger ", i)

        if sts == gs.PARAMETERS_MISSING:
            missing_parameters = True

    if missing_parameters:
        print("One or more trigger parameters missing, using defaults for missing values")                        

    for i in range(system_info['BoardCount']):
        g_cardTotalData.append(0)
		
    return status
    
    
def initialize_stream(handle):
    expert_options = CS_BBOPTIONS_STREAM
    acq = PyGage.GetAcquisitionConfig(handle)
    if not isinstance(acq, dict):
        if not acq:
            print("Error in call to GetAcquisitionConfig")
            return CS_MISC_ERROR
        else: # should be error code
            print("Error: ", PyGage.GetErrorString(acq))
            return acq
    
    extended_options = PyGage.GetExtendedBoardOptions(handle)
    if extended_options < 0:
        print("Error: ", PyGage.GetErrorString(extended_options))
        return extended_options
        
    if extended_options & expert_options:
        print("\nSelecting Expert Stream from image 1.")
        acq['Mode'] |= CS_MODE_USER1
    elif (extended_options >> 32) & expert_options:
        print("\nSelecting Expert Stream from image 2.")    
        acq['Mode'] |= CS_MODE_USER2
    else:
        print("\nCurrent system does not support Expert Stream.")
        print("\nApplication terminated")
        return CS_MISC_ERROR
        
    status = PyGage.SetAcquisitionConfig(handle, acq)
    if status < 0:
        print("Error ", PyGage.GetErrorString(status))
    return status
    

def load_stm_configuration(filename):
    app = {}
    # set reasonable defaults

    app['SaveToFile'] = 0
    app['TimeoutOnTransfer'] = TRANSFER_TIMEOUT
    app['FileFlagNoBuffering'] = 1
    app['BufferSize'] = STREAM_BUFFERSIZE
    app['DataFile'] = OUT_FILE
    app['ErrorHandlingMode'] = 1
    app['DataPackMode'] = 0
    
    config = ConfigParser()

    # parse existing file
    config.read(filename)
    section = 'StmConfig'    
    
    if section in config:
        for key in config[section]:
            key = key.lower()
            value= config.get(section, key)
            if key == 'savetofile':	
                app['SaveToFile'] = int(value)
            elif key == 'timeoutontransfer':
                app['TimeoutOnTransfer'] = int(value)
            elif key == 'fileflagnobuffering':
                app['FileFlagNoBuffering'] = int(value)
            elif key == 'buffersize': # in bytes
                app['BufferSize'] = int(value) # may need to be an int64
            elif key == 'errorhandlingmode':
                app['ErrorHandlingMode'] = int(value)
            elif key == 'datapackmode':
                app['DataPackMode'] = int(value)
            elif key == 'datafile':
                app['DataFile'] = value
    return app
    
  
def initialize():
    status = PyGage.Initialize()
    if status < 0:
        return status
    else:
        handle = PyGage.GetSystem(0, 0, 0, 0)
        return handle		
            

def main():
    inifile = 'Stream2Disk.ini'
    handle = initialize()
    if handle < 0:
        # get error string
        error_string = PyGage.GetErrorString(handle)
        print("Error: ", error_string)
        raise SystemExit
		
    system_info = PyGage.GetSystemInfo(handle)
    if not isinstance(system_info, dict): # if it's not a dict, it's an int indicating an error
        print("Error: ", PyGage.GetErrorString(system_info))
        PyGage.FreeSystem(handle)
        raise SystemExit

    print("\nBoard Name: ", system_info["BoardName"])    

        
    status = configure_system(handle, inifile)

    if status < 0:
        # get error string
        error_string = PyGage.GetErrorString(status)
        print("Error: ", error_string)
        PyGage.FreeSystem(handle)
        raise SystemExit
        
    app = load_stm_configuration(inifile)

    status = initialize_stream(handle)

    if status < 0:
        # The error string is printed out in initialize_stream
        PyGage.FreeSystem(handle)
        raise SystemExit
    
    # commit after initialize_stream so changes take effect
    packed_stream_supported = False
    status = PyGage.GetDataPackingMode(handle)
    if status < 0:
        if CS_INVALID_PARAMS_ID == status:
            if app['DataPackMode'] != 0:
                print("\nThe current CompuScope system does not support "
                      "Data Streaming in Packed Mode\n")
                PyGage.FreeSystem(handle)
                raise SystemExit
        else:
            print("Error: ", PyGage.GetErrorString(status))
            PyGage.FreeSystem(handle)
            raise SystemExit
    else: # The system supports streaming in packed data mode, so set the mode before committing
        packed_stream_supported = True
        status = PyGage.SetDataPackingMode(handle, app['DataPackMode'])
        if status < 0:
            # get error string
            error_string = PyGage.GetErrorString(status)
            print("Error: ", error_string)
            PyGage.FreeSystem(handle)
            raise SystemExit
    
    
    status = PyGage.Commit(handle)   
    
    if status < 0:    
        # get error string
        error_string = PyGage.GetErrorString(status)
        print("Error: ", error_string)
        PyGage.FreeSystem(handle)
        raise SystemExit
    
    # after commit the sample size may change
    acq_config = PyGage.GetAcquisitionConfig(handle) # check for error

    if packed_stream_supported:
        dataFormat = PyGage.GetDataFormatInfo(handle)
        if not isinstance(dataFormat, dict):
            print("Error: ", PyGage.GetErrorString(dataFormat))
            PyGage.FreeSystem(handle)
            raise SystemExit
    
    # get total amount of data we expect to receive
    total_samples = PyGage.GetStreamTotalDataSizeInBytes(handle)
    
    if total_samples < 0 and total_samples != acq_config['SegmentSize']:
        print("Error: ", PyGage.GetErrorString(total_samples))
        PyGage.FreeSystem(handle)
        raise SystemExit
        
    # convert to samples
    if total_samples != -1:
        total_samples = total_samples // system_info['SampleSize']
        # we're using system_info["SampleSize"] because we're using datapacking mode
        # which can change the sample size in acq_config["SampleSize"]

    threads = []
    stream_started_event = threading.Event()
    ready_for_stream_event = threading.Event()
    stream_aborted_event = threading.Event()
    stream_error_event = threading.Event()
    
    for i in range(system_info['BoardCount']):
        card_index = i+1
        t = threading.Thread(target=save_data_to_file, args=(handle, card_index, 
                                                             system_info['SampleSize'],
                                                             app, stream_started_event,
                                                             ready_for_stream_event,
                                                             stream_aborted_event,
                                                             stream_error_event))
        threads.append(t)
        t.start()
        set = ready_for_stream_event.wait(5)
        if not set:
            print("\nThread initialization error on card ", card_index)
            stream_aborted_event.set()
            PyGage.FreeSystem(handle)
            raise SystemExit

    print("\nStarting streaming. Press CTRL-C to abort\n\n")
        
    status = PyGage.StartCapture(handle)     
    if status < 0:    
        # get error string
        print("Error: ", PyGage.GetErrorString(status))
        PyGage.FreeSystem(handle)
        raise SystemExit
            
    # get tick count
    tickStart = time.time()
    stream_started_event.set()
        
    main_thread = threading.current_thread()
    
    Done = False
    aborted = False
    error_occurred = False
    try:
        while not Done:
            set = stream_error_event.wait(0.5) # arbitrary amount of time
            if set: # error occured
                error_occurred = True
                done = True
            for t in threading.enumerate():
                tickNow = time.time()
                systemTotalData = sum(g_cardTotalData)
                if t is not main_thread:
                    t.join(0.3) # timeout of 0.3 seconds

            systemTotalData = sum(g_cardTotalData)
            update_progress(tickNow - tickStart, systemTotalData * system_info['SampleSize'])

            count = 0
            for t in threading.enumerate():
                if t is not main_thread:
                    count += 1
            if count == 0:
                Done = True

    except KeyboardInterrupt:
        stream_aborted_event.set()
        aborted = True
        # wait for all threads to end
        for i in threads:
            i.join()                

    PyGage.AbortCapture(handle)
    PyGage.FreeSystem(handle)
      
    if error_occurred:
        print("\nStream aborted on error\n")
    elif aborted:
        print("\nStream aborted by user\n")
    else:
        print("\nStream has finished {} segments\n".format(acq_config['SegmentCount']))

    total_data = (1.0 * systemTotalData) / 1000000

    
    if packed_stream_supported:
        unpacked_size = total_data * system_info['SampleSize'] * 8 / dataFormat['SampleSizeBits']
        print("\nTotal Data in '{0}-bit' samples: {1:0.2f} Ms\n".format(dataFormat['SampleSizeBits'], unpacked_size))
    else:
        print("\nTotal data in '{0}-bit' samples: {1:0.2f} MS\n".format((8 * system_info['SampleSize']), total_data))


if __name__ == '__main__':
    main()

