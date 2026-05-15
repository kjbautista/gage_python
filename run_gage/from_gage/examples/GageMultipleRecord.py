from __future__ import print_function
from builtins import int
import platform
import sys
from datetime import datetime
import GageSupport as gs
import GageConstants as gc
import platform

# Code used to determine if python is version 2.x or 3.x
# and if os is 32 bits or 64 bits.  If you know they
# python version and os you can skip all this and just
# import the appropriate version

# returns is_64bits for python 
# (i.e. 32 bit python running on 64 bit windows should return false)

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


def configure_system(handle, filename):
    acq, sts = gs.LoadAcquisitionConfiguration(handle, filename)	

    if isinstance(acq, dict) and acq:	
        status = PyGage.SetAcquisitionConfig(handle, acq)
        if status < 0:
            return status
    else:
        print("Using defaults for acquisition parameters")

    if sts == gs.INI_FILE_MISSING:
        print("Missing ini file, using defaults")
    elif sts == gs.PARAMETERS_MISSING:
        print("One or more acquisition parameters missing, using defaults for missing values")        
 
    system_info = PyGage.GetSystemInfo(handle)
    acq = PyGage.GetAcquisitionConfig(handle) # check for error - copy to GageAcquire.py

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
		
    status = PyGage.Commit(handle)
    return status
        

def initialize():
    status = PyGage.Initialize()
    if status < 0:
        return status
    else:
        handle = PyGage.GetSystem(0, 0, 0, 0)
        return handle		
        
def transfer_time_stamp(handle, start, count):
    timestamps = PyGage.TransferData(handle, 1, gc.TxMODE_TIMESTAMP, 1, start, count)
    if isinstance(timestamps, int): # an error occurred
        print("\nError getting time stamp data")
        return timestamps
    # The tick_frequency is the clock rate of the counter used to acquire the time stamp data
    tick_frequency = PyGage.GetTimeStampFrequency(handle)
    if tick_frequency < 0:
        print("\nError getting tick frequency")
        return tick_frequency
    return timestamps[0], tick_frequency


def save_data_to_file(handle, mode, app, system_info):
    status = PyGage.StartCapture(handle)
    if status < 0:
        return status
        
    capture_time = 0        
    status = PyGage.GetStatus(handle)    
    while status != gc.ACQ_STATUS_READY:
        status = PyGage.GetStatus(handle)
        # if we've triggered, get the time of day
        # this is just to demonstrate how to use the time stamp 
        # in the SIG file header
        if status == gc.ACQ_STATUS_TRIGGERED:
            capture_time = datetime.now().time() 

    # just in case we missed the trigger time, we'll use the capture time
    if capture_time == 0: 
        capture_time = datetime.now().time()
    
    channel_increment = gs.CalculateChannelIndexIncrement(mode, 
                                                          system_info['ChannelCount'], 
                                                          system_info['BoardCount'])

    acq = PyGage.GetAcquisitionConfig(handle)
    # These fields are common for all the channels

    # Validate the start address and the length. This is especially 
    # necessary if trigger delay is being used.

    min_start_address = acq['TriggerDelay'] + acq['Depth'] - acq['SegmentSize']
    if app['StartPosition'] < min_start_address:
        print("\nInvalid Start Address was changed from {0} to {1}".format(app['StartPosition'],  min_start_address))
        app['StartPosition'] = min_start_address

    max_length = acq['TriggerDelay'] + acq['Depth'] - min_start_address
    if app['TransferLength'] > max_length:
        print("\nInvalid Transfer Length was changed from {0} to {1}".format(app['TransferLength'], max_length))
        app['TransferLength'] = max_length

    
    stHeader = {}
    if acq['ExtClk']:
        stHeader['SampleRate'] = acq['SampleRate'] / (acq['ExtClkSampleSkip'] * 1000)
    else:
        stHeader['SampleRate'] = acq['SampleRate'] 
    
    stHeader['Start'] = app['StartPosition']
    stHeader['Length'] = app['TransferLength']
    stHeader['SampleSize'] = acq['SampleSize']
    stHeader['SampleOffset'] = acq['SampleOffset']
    stHeader['SampleRes'] = acq['SampleResolution']
    stHeader['SampleBits'] = acq['SampleBits']
    
    if app['SaveFileFormat'] == gs.TYPE_SIG:
        stHeader['SegmentCount'] = 1
    else:
        stHeader['SegmentCount'] = acq['SegmentCount']
    
    # if we're saving a txt file, get all the timestamps now
        if app['SaveFileFormat'] in [gs.TYPE_DEC, gs.TYPE_HEX, gs.TYPE_FLOAT]:
            timestamps, frequency = transfer_time_stamp(handle, app['SegmentStart'], app['SegmentCount'])
            time_stamp_data = [i * 1000000 / frequency for i in timestamps.tolist()]

#    count = 0
    for i in range(1, system_info['ChannelCount'] + 1, channel_increment):
        count = 0
        for group in range(app['SegmentStart'], app['SegmentStart'] + app['SegmentCount']):
            buffer = PyGage.TransferData(handle, i, gc.TxMODE_DEFAULT, group, app['StartPosition'], app['TransferLength'])
            if isinstance(buffer, int): # an error occurred
                print("Error transferring channel ", i)
                return buffer

            # if call succeeded (buffer is not an integer) then
            # buffer[0] holds the actual data, buffer[1] holds
            # the actual start and buffer[2] holds the actual length                    
            
            chan = PyGage.GetChannelConfig(handle, i)        
            stHeader['InputRange'] = chan['InputRange']
            stHeader['DcOffset'] = chan['DcOffset']
            stHeader['SegmentNumber'] = group

            if app['SaveFileFormat'] == gs.TYPE_SIG:
                filename = app['SaveFileName'] + '_CH' + str(i) + '_' + str(group) + '.sig'
            elif app['SaveFileFormat'] == gs.TYPE_BIN:
                filename = app['SaveFileName'] + '_CH' + str(i) + '_' + str(group) + '.dat'
            else:
                filename = app['SaveFileName'] + '_CH' + str(i) + '_' + str(group) + '.txt'        

            # TransferData may change the actual length of the buffer 
            # (i.e. if the requested transfer length was too large), so we can
            # change it in the header to be the length of the buffer or
            # we can use the actual length (buffer[2])
            stHeader['Length'] = buffer[2]        

            if app['SaveFileFormat'] in [gs.TYPE_DEC, gs.TYPE_HEX, gs.TYPE_FLOAT]:
                timeStamp = time_stamp_data[count]
                count += 1
            else:
                timeStamp = {}
                timeStamp['Hour'] = capture_time.hour
                timeStamp['Minute'] = capture_time.minute
                timeStamp['Second'] = capture_time.second
                timeStamp['Point1Second'] = capture_time.microsecond // 1000 # convert to milliseconds

            stHeader['TimeStamp'] = timeStamp
            status = gs.SaveFile(filename, i, buffer[0], app['SaveFileFormat'], stHeader)

    return status	
    
		
def main():
    inifile = 'MultipleRecord.ini'
    try:
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
        else:
            acq_config = PyGage.GetAcquisitionConfig(handle)
            app, sts = gs.LoadApplicationConfiguration(inifile)	

            # we don't need to check for gs.INI_FILE_MISSING because if there's no ini file
            # we've already reported when calling configure_system
            if sts == gs.PARAMETERS_MISSING:
                print("One or more application parameters missing, using defaults for missing values")

            status = save_data_to_file(handle, acq_config['Mode'], app, system_info)
            if isinstance(status, int):
                if status < 0:
                    error_string = PyGage.GetErrorString(status)
                    print("Error: ", error_string)
                elif status == 0:  # could not open o write to the data file
                    print("Error opening or writing ", filename)
                else:
                    if app['SaveFileFormat'] == gs.TYPE_SIG:
                        print("\nAcquisition completed.\nAll channels saved as "
                              "GageScope SIG files in the current directory\n")
                    elif app['SaveFileFormat'] == gs.TYPE_BIN:
                        print("\nAcquisition completed.\nAll channels saved "
                              "as binary files in the current directory\n")
                    else:
                        print("\nAcquisition completed.\nAll channels saved "
                             "as ASCII data files in the current directory\n")
            else: # not an int, we can't open or write the file so we returned the filename
                print("Error opening or writing ", status)
    except  KeyboardInterrupt:
        print("Exiting program")

    PyGage.FreeSystem(handle) 
    
   
if __name__ == '__main__':
    main()
    
