from __future__ import print_function, division
from builtins import int
# configparser is needed to read ini files
try:
    from configparser import ConfigParser # needed to read ini files
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0

import threading
import sys
import time
import itertools # for infinite for loop
from collections import namedtuple

import GageSupport as gs
import numpy as np

from GageConstants import (CS_CURRENT_CONFIGURATION, CS_ACQUISITION_CONFIGURATION,
                           CS_STREAM_TOTALDATA_SIZE_BYTES, CS_DATAPACKING_MODE, CS_MASKED_MODE,
                           CS_GET_DATAFORMAT_INFO, CS_BBOPTIONS_STREAM, CS_MODE_USER1,
                           CS_MODE_USER2, CS_EXTENDED_BOARD_OPTIONS, STM_TRANSFER_ERROR_FIFOFULL,
                           CS_SEGMENTTAIL_SIZE_BYTES, CS_TIMESTAMP_TICKFREQUENCY) 

from GageErrors import (CS_MISC_ERROR,
                        CS_INVALID_PARAMS_ID,
                        CS_STM_TRANSFER_TIMEOUT,
                        CS_STM_COMPLETED)

import array

# Code used to determine if python is version 2.x or 3.x
# and if os is 32 bits or 64 bits.  If you know they
# python version and os you can skip all this and just
# import the appropriate version

import platform

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

MAX_SEGMENT_COUNT = 25000

g_segmentCounted = []
g_cardTotalData = []
g_tickFrequency = 0


class StreamInfo:
    __slots__ = ['WorkBuffer', 'TimeStamp', 'BufferSize', 'SegmentSize', 'TailSize',
                 'LeftOverSize', 'BytesToEndSegment', 'BytesToEndTail', 'DeltaTime',
                 'LastTimeStamp', 'Segment', 'SegmentCountDown', 'SplitTail']
    pass



def save_results(stream_info, processed_segments, segment_file_count,
                file_count, f, filename, card_index,
                total_segment_count, sample_rate):

    global g_tickFrequency

    # if the file is full (> MAX_SEGMENT_COUNT), open  new file
    for i in range(processed_segments):
        if segment_file_count > MAX_SEGMENT_COUNT:
            f, file_count = update_result_file(f, filename,  card_index, file_count)
            segment_file_count = 0
    
        # we're casting to an int here for Python 2. There isn't
        # a 64 bit int type in for arrys in Python 2 so our 
        # TimeStamp array are doubles so we need to cast it here.
        # In Python 3 the array is 64 bit integers so there's no
        # need to cast.
        TsLow = int(stream_info.TimeStamp[i]) & 0xffffffff
        TsHigh = int(stream_info.TimeStamp[i]) >> 32            
        
        stream_info.DeltaTime = stream_info.TimeStamp[i] - stream_info.LastTimeStamp
        delta_time = 1000.0 * stream_info.DeltaTime / g_tickFrequency
        delta = sample_rate * delta_time // 1000

        if stream_info.Segment == 1:
            s = "{0}\t\t{1:08x} {2:08x}\r\n".format(total_segment_count, TsHigh, TsLow)
            f.write(s)
        else:
            s = "{0}\t\t{1:08x} {2:08x}\t{3:0.5f}\t\t{4:0.0f} \r\n".format(total_segment_count,
                                                              TsHigh, TsLow, delta_time, delta)
            f.write(s)

        total_segment_count += 1
        stream_info.LastTimeStamp = stream_info.TimeStamp[i]
        segment_file_count += 1
        stream_info.Segment += 1        
    return f, segment_file_count, file_count, total_segment_count


def update_result_file(f, filename, card_index, file_count):
    global g_tickFrequency

    file_count += 1
    if f != None:
        f.close()

    name = filename + "_" + str(card_index) + "_" + str(file_count) + ".txt"
    f = open(name, "w")
    s = "Timestamp Frequency = {}\n".format(g_tickFrequency)

    f.write(s)
    s = "Segment start with segment offset: {} x 25000 segments per file\n\n".format(file_count-1)
    f.write(s)
    s = "Segment\t\tTimestamp (H/L)\t\tDelta(ms)\tDelta(Samples)\n"
    f.write(s)
    return f, file_count



def array_to_time_stamp(arr):
    x = 0
    # need to convert to list because leaving it as a numpy array causes values
    # greater than 2**32 to be treated as negative
    a = arr.tolist()  
    for i in range(len(arr)):
        x = x | (a[i] << (i * 8))

    if sys.version_info > (3, 0):
        return x
    else:
        return float(x)


def analysis_func(stream_info, buffer_size_in_bytes):
    # The time stamp is the first 8 bytes (int64) of the tail
    global g_segmentCounted
    
    bytes_to_buffer_end = buffer_size_in_bytes
    buffer_start = 0
    index = 0
    
    del stream_info.TimeStamp[:] # clear out Time Stamp array

    # If tail was split in last buffer
    if stream_info.SplitTail:
        #  Save leftOver from last buffer 
        stream_info.LeftOverSize = stream_info.BytesToEndTail
        bytes_to_buffer_end -= stream_info.LeftOverSize

        # Goto next segment 
        buffer_start = stream_info.BytesToEndTail

        stream_info.BytesToEndTail = stream_info.TailSize

        # Reset flag for tail split 
        stream_info.SplitTail = False

    # case Buffer < Segment 
    if bytes_to_buffer_end < (stream_info.SegmentSize + stream_info.TailSize):
        for i in itertools.count():
            # if there is a tail in the present segment 
            if bytes_to_buffer_end >= (stream_info.BytesToEndSegment + stream_info.BytesToEndTail):
                # Goto Tail 
                bytes_to_buffer_end -= stream_info.BytesToEndSegment

                # Process Tail
                buffer_start += stream_info.BytesToEndSegment
                buffer_end = buffer_start + 6
                arr = stream_info.WorkBuffer[buffer_start:buffer_end]
                stream_info.TimeStamp.append(array_to_time_stamp(arr))
                index += 1
           
                stream_info.SegmentCountDown -= 1
                bytes_to_buffer_end -= stream_info.BytesToEndTail

                # Reset counts 
                stream_info.BytesToEndSegment = stream_info.SegmentSize
                stream_info.BytesToEndTail = stream_info.TailSize

                if not stream_info.SegmentCountDown:
                    return index

            # If there is only part of a tail in the present segment 
            elif bytes_to_buffer_end  > stream_info.BytesToEndSegment:
                # Goto Tail 
                bytes_to_buffer_ends -= stream_info.BytesToEndSegment

                # Process part tail 
                buffer_start += stream_info.BytesToEndSegment
                buffer_end = buffer_start + 6
                
                arr = stream_info.WorkBuffer[buffer_start:buffer_end]
                stream_info.TimeStamp.append(array_to_time_stamp(arr))
                index += 1

                stream_info.SegmentCountDown -= 1
                stream_info.BytesToEndTail -= bytes_to_buffer_end

                # Tail is split between this buffer and next one 
                stream_info.plitTail = True
                stream_info.BytesToEndSegment = stream_info.SegmentSize
                if not stream_info.SegmentCountDown:
                    return index
                break
			
            stream_info.BytesToEndSegment -= bytes_to_buffer_end
            break


    # Case Buffer >= Segment 
    else:
        for i in itertools.count():
            # Goto tail 
            buffer_start += stream_info.BytesToEndSegment
            buffer_end = buffer_start + 6
            bytes_to_buffer_end -= stream_info.BytesToEndSegment

            # Process Tail
            arr = stream_info.WorkBuffer[buffer_start:buffer_end]
            stream_info.TimeStamp.append(array_to_time_stamp(arr))
            index += 1

            stream_info.SegmentCountDown -= 1

            # goto next segment 

            buffer_start += stream_info.BytesToEndTail
            buffer_end = buffer_start + 6
            bytes_to_buffer_end -= stream_info.BytesToEndTail

            # Reset counts 
            stream_info.BytesToEndSegment = stream_info.SegmentSize
            stream_info.BytesToEndTail = stream_info.TailSize

            # If we reach the end of the segments to analyse 
            if not stream_info.SegmentCountDown:
                return index

            # No more tail in buffer
            if bytes_to_buffer_end <= stream_info.BytesToEndSegment:
                stream_info.BytesToEndSegment -= bytes_to_buffer_end
                stream_info.LeftOverSize = 0
                break
            else: # No  more full tail in buffer
                if bytes_to_buffer_end < (stream_info.BytesToEndSegment + 
                                          stream_info.BytesToEndTail):
                    # jump to Tail location

                    buffer_start += stream_info.BytesToEndSegment
                    buffer_end = buffer_start + 6
                    bytes_to_buffer_end -= stream_info.BytesToEndSegment

                    # Process part of tail
                    arr = stream_info.WorkBuffer[buffer_start:buffer_end]
                    stream_info.TimeStamp.append(array_to_time_stamp(arr))
                    index += 1

                    stream_info.SegmentCountDown -= 1
                    stream_info.BytesToEndTail -= bytes_to_buffer_end - stream_info.LeftOverSize
					
                    #  Tail is split between this buffer and next one
                    stream_info.SplitTail = True

                    if not stream_info.SegmentCountDown:
                        return index
                    break

    return index


def card_stream(handle, card_index, sample_size, app,
               stream_started_event, ready_for_stream_event,
               stream_aborted_event, stream_error_event):
    global g_segmentCounted
    global g_cardTotalData

    f = None
    buffer1 = PyGage.GetStreamingBuffer(handle, card_index, app['BufferSize'])
    if isinstance(buffer1, int):  # in python 2 would be isinstance(buffer1, (int, long))
        print("Error getting streaming buffer 1: ", PyGage.GetErrorString(buffer1))
        stream_error_event.set()
        time.sleep(1) # to give stream_error_wait() a chance to catch it
        return False

    buffer2 = PyGage.GetStreamingBuffer(handle, card_index, app['BufferSize'])
    if isinstance(buffer2, int):
        print("Error getting streaming buffer 2: ", PyGage.GetErrorString(buffer2))
        PyGage.FreeStreamingBuffer(handle, card_index, buffer1)
        stream_error_event.set()
        time.sleep(1) # to give stream_error_wait() a chance to catch it        
        return False

    acq = PyGage.GetAcquisitionConfig(handle)

    stream_info = StreamInfo()
    
    data_in_segment_samples = acq['SegmentSize'] * (acq['Mode'] & CS_MASKED_MODE)

    status = PyGage.GetSegmentTailSizeInBytes(handle)
    if status < 0:
        print("Error: ", PyGage.GetErrorString(status))
        return

    segment_tail_size_in_bytes = status

    segment_size_in_bytes = data_in_segment_samples * sample_size
    transfer_size_in_samples = app['BufferSize'] // sample_size
    print("\nActual buffer size used for data streaming = ", app['BufferSize'])

    segment_per_buffer = app['BufferSize'] // ((acq['SegmentSize'] * acq['SampleBits'] // 8) + 64)

    ready_for_stream_event.set()

    stream_started_event.wait() # should also be waiting for abort
    done = False
    stream_completed_success = False
    loop_count = 0
    total_data = 0
    work_buffer_active = False
    tail_left_over = 0
    segment_file_count = 1
    segment_count = 0
    file_count = 0
    total_segment_count = 1

    # Initialize the first result file
    f, file_count = update_result_file(f, app['ResultsFile'], card_index, file_count)
    
    buffer = np.zeros_like(buffer1)
    
    stream_info.WorkBuffer = np.zeros_like(buffer1)
    try: # Can also just check if the python  version >= 3
        stream_info.TimeStamp = array.array('q')
    except ValueError:
        stream_info.TimeStamp = array.array('d')        

    stream_info.BufferSize = app['BufferSize']
    stream_info.SegmentSize = segment_size_in_bytes
    stream_info.TailSize = segment_tail_size_in_bytes
    stream_info.BytesToEndSegment = segment_size_in_bytes
    stream_info.BytesToEndTail = segment_tail_size_in_bytes
    stream_info.LeftOverSize = tail_left_over
    stream_info.LastTimeStamp = 0
    stream_info.Segment = 1
    stream_info.SegmentCountDown = acq['SegmentCount']
    stream_info.SplitTail = False

    while not done and not stream_completed_success:
        set = stream_aborted_event.wait(0)
        if set: # user has aborted
            break
        # on any error, set an error event
        if loop_count & 1:
            buffer = buffer2
        else:
            buffer = buffer1

        status = PyGage.TransferStreamingData(handle, card_index, buffer, transfer_size_in_samples)

        if status < 0:
            if status == CS_STM_COMPLETED:
                stream_completed = True
            else:
                print("Error: ", PyGage.GetErrorString(status))
                stream_error_event.set()
                time.sleep(1) # to give stream_error_wait() a chance to catch it                
                break

        if app.get('DoAnalysis'):  # does not raise KeyError if key does not exist
            if work_buffer_active:
                processed_segments = analysis_func(stream_info, app.get('BufferSize'))
                if processed_segments > 0:
                    f, segment_file_count, file_count, total_segment_count = save_results(stream_info, 
                                                                                          processed_segments, 
                                                                                          segment_file_count, 
                                                                                          file_count, 
                                                                                          f, 
                                                                                          app['ResultsFile'], 
                                                                                          card_index, 
                                                                                          total_segment_count,
                                                                                          acq['SampleRate'])

                g_segmentCounted[card_index-1] += processed_segments
                    
        # Wait for the DMZ transfer on the current bufer to complete
        # so we can loop back around to start a new one. Calling thread
        # will sleep until the trnasfer completes

        p = PyGage.GetStreamingTransferStatus(handle, card_index, app['TimeoutOnTransfer'])
        if isinstance(p, tuple):
            g_cardTotalData[card_index-1] += p[1]  # have total_data be an array, 1 for each card            
            if p[2] == 0:
                stream_completed_success = False
            else:
                stream_completed_success = True

            if STM_TRANSFER_ERROR_FIFOFULL & p[0]:
                error_handling = 1
                if error_handling != 0:
                    print("Fifo full detected on card ", card_index)
                    done = True                
        else: # error detected 
            done = True
            if p == CS_STM_TRANSFER_TIMEOUT:
                print("\nStream transfer timeout on card ", card_index)
            else:
                print("5 Error: ", p)
                print("5 Error: ", PyGage.GetErrorString(p))

        stream_info.WorkBuffer = buffer
        work_buffer_active = True
        loop_count = loop_count + 1

    if app.get('DoAnalysis'): 
        # does not raise KeyError if key doesn't exist. Using app['DoAnalysis'] does raise KeyError
        # Do analysis on last buffer
        processed_segments = analysis_func(stream_info, app.get('BufferSize'))
        f, segment_file_count, file_count, total_segment_count = save_results(stream_info, 
                                                                              processed_segments, 
                                                                              segment_file_count, 
                                                                              file_count, 
                                                                              f, 
                                                                              app['ResultsFile'], 
                                                                              card_index, 
                                                                              total_segment_count,
                                                                              acq['SampleRate'])

        g_segmentCounted[card_index-1] += processed_segments
        if f is not None:
            f.close()
    
    status = PyGage.FreeStreamingBuffer(handle, card_index, buffer1)
    status = PyGage.FreeStreamingBuffer(handle, card_index, buffer2)
    if stream_completed_success:
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
        s = "Total: {0:.2f} MB, Rate: {1:6.2f} MB/s Elapsed time: {2:d}:{3:02d}:{4:02d}\r".format(
                                                                                           total, 
                                                                                           rate, 
                                                                                           hours, 
                                                                                           minutes, 
                                                                                           seconds)
        sys.stdout.write(s)
        sys.stdout.flush()


def configure_system(handle, filename):
    global g_segmentCounted
    global g_cardTotalData
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

    channel_increment = gs.CalculateChannelIndexIncrement(acq['Mode'], system_info['ChannelCount'],
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
        g_segmentCounted.append(0)

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
        print("\nSelecting Expert Stream from image 1")
        acq['Mode'] |= CS_MODE_USER1
    elif (extended_options >> 32) & expert_options:
        print("\nSelecting Expert Stream from image 2")
        acq['Mode'] |= CS_MODE_USER2
    else:
        print("\nCurrent system does not support Expert Streaming")
        print("\nApplication terminated")
        return CS_MISC_ERROR

    status = PyGage.SetAcquisitionConfig(handle, acq)
    if status < 0:
        print("Error: ", PyGage.GetErrorString(status))
    return status


def load_stm_configuration(filename):
    app = {}
    # set reasonable defaults

    app['TimeoutOnTransfer'] = TRANSFER_TIMEOUT
    app['BufferSize'] = STREAM_BUFFERSIZE
    app['DoAnalysis'] = 0
    app['ResultsFile'] = 'Result'

    config = ConfigParser()

    # parse existing file
    config.read(filename)
    section = 'StmConfig'

    if section in config:
        for key in config[section]:
            key = key.lower()
            value= config.get(section, key)
            if key == 'doanalysis':
                if int(value) == 0:
                    app['DoAnalysis'] = False
                else:
                    app['DoAnalysis'] = True
            elif key == 'timeoutontransfer':
                app['TimeoutOnTransfer'] = int(value)
            elif key == 'buffersize': # in bytes
                app['BufferSize'] = int(value) # may need to be an int64
            elif key == 'resultsfile':
                app['ResultsFile'] = value
    return app


def initialize():
    status = PyGage.Initialize()
    if status < 0:
        return status
    else:
        handle = PyGage.GetSystem(0, 0, 0, 0)
        return handle


def main():
    global g_tickFrequency
    global g_segmentCounted
    global g_cardTotalData
    inifile = 'Stream2Analysis.ini'
    handle = initialize()
    if handle < 0:
        # get error string
        error_string = PyGage.GetErrorString(handle)
        print("Error: ", error_string)
        raise SystemExit # ??

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
        raise SystemExit # ??

    app = load_stm_configuration(inifile)

    status = initialize_stream(handle)

    if status < 0:
        # error string is printed out in initialize_stream
        PyGage.FreeSystem(handle)
        raise SystemExit
    
    status = PyGage.Commit(handle)

    if status < 0:
        # get error string
        error_string = PyGage.GetErrorString(status)
        print("Error: ", error_string)
        PyGage.FreeSystem(handle)
        raise SystemExit

    g_tickFrequency = PyGage.GetTimeStampFrequency(handle)

    if g_tickFrequency < 0:
        print("Error: ",  PyGage.GetErrorString(g_tickFrequency))
        PyGage.FreeSystem(handle)
        raise SystemExit 

    # after commit the sample size may change
    acq_config = PyGage.GetAcquisitionConfig(handle) 

    # get total amount of data we expect to receive
    total_samples = PyGage.GetStreamTotalDataSizeInBytes(handle)
    
    if total_samples < 0 and total_samples != acq_config['SegmentSize']:
        print("Error: ", PyGage.GetErrorString(total_samples))
        PyGage.FreeSystem(handle)
        raise SystemExit

    # convert to samples
    if total_samples != -1:
        total_samples = total_samples // system_info['SampleSize']

    threads = []
    stream_started_event = threading.Event()
    ready_for_stream_event = threading.Event()
    stream_aborted_event = threading.Event()
    stream_error_event = threading.Event()

    for i in range(system_info['BoardCount']):
        card_index = i+1
        t = threading.Thread(target=card_stream, args=(handle, card_index, 
                                                       system_info['SampleSize'],
                                                       app, 
                                                       stream_started_event, 
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
        raise SystemExit # ??

    # get tick count
    tickStart = time.time()
    stream_started_event.set()

    main_thread = threading.current_thread()

    Done = False
    aborted = False
    error_occurred = False
    try:
        while not Done:
            set = stream_error_event.wait(0.5)
            if set: # error occured
                error_occured = True
                donw = True
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
        if g_segmentCounted[0] != acq_config['SegmentCount']:
            print("\nStream has finished with {0} loops instead of {1}\n".format(g_segmentCounted[0],
                                                                          acq_config['SegmentCount']))
        else:
            print("\nStream has finished with {} loops\n".format(acq_config['SegmentCount']))



if __name__ == '__main__':
    main()

