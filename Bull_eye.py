# Copyright (C) 2022 - present, AMCG Inc. Ltd. written by <WonSik.Jung> <jws@amcg.co.kr>
#
# calculate 3D egg model_mapping Activity


'''    Parameters
    ----------
    INP : heart,leadfield
        the Data Structure list of KRISSsoftware

    Read_INP :  "np.ndarray"
        the Function read list of heart position, leadfiled

    Vertex: "np.ndarray"
        position of xyz coordinate in heart.inp

    Element: "np.array"
        Triangle infomation how to connect Vertex in heart.inp

    Moved_pos: "np.array"
        the vertex move their center to origin(0,0,0)s

    dataset: "dictonary file","Time series data in kdf"
        the dictonary file key(channel,data)

    reduced lead file: "np.array"
        lead file actived channel index





    mesh:"open3d.geometry.TriangleMesh()"
        mesh composed of Vertex,VertexColor,Triangle
        when you put component, it automatically caculate polygon Structure

    mesh.Vertices: "open3d.utility"[vertex number*3]
        the mesh position in xyz coordinate

    mesh.vertex_colors: "open3d.utility"[vertex number*3]
        the color of Vertex in jet scale

    mesh.triangles:"open3d.utility.vector"
        the mesh combination how to connect Vertex [N*3]

    Frontal:"open3d.geometry.TriangleMesh()"
        The structment heart.inp position data

    Lateral:"open3d.geometry.TriangleMesh()"
        The structment hert.inp position to Lateral

    Inferior:"open3d.geometry.TriangleMesh()"
        The structment hert.inp position to Inferior



    ---------Manual

    1) setting_3D_Vertex(heart_inp_path)

    Return: 3D Vertex "np.array"
        raise: When heart.inp path is not corrected

    2) Read_source_amp(kdf_dataes,lead_file,timecursor):

        2-1)kdf_dataes:[Channels* recording time]
            Time Series Data After Handing

        2-2)lead_file:"lead_field_path"
            lead_field:[Channels, Heart_Vertex*3]
            Describe Source Activity , Depend on distance between heart and sensor

        2-3)timecursor:"int"
            extract time when Recording time of Kdf_dataes

    Return: 2D Activity Norm normalized
    # We Loaded Lab View Source Norm, To check match visualization Result

    3)visulize_Egg(source_3d_norm,axis=None)

        3-1)source_3d_norm:"numpy.array"
            Return Result on (2)

        3-2)axis:"str"
            axis_list=[['Frontal','F','1'],['Lateral','L','2'],['Inferior','I','3']]
            axis: Visuailzing Direction, default visualizing All Direction


        example:
        visualize_Egg(source_3d)
        visualize_Egg(source_3d,'F')

    Return : "Open3d.visualize(mesh)"
    ------

    Note
    ----
    See the following articles for understanding theoretical backgrounds.
        Jia-Zhu Wang, et al., Magnetic Source Image Determined by a Lead_Field Analysis: The Unique Minimun-Norm Least-Squares Estimation(1992)
        Jukka T.Nenonen, et al., Solving the Inverse Problem in Magnetocardiography(1994)'''

import os
import math as m
import numpy as np
import matplotlib.pylab as plt
from numpy import linalg
from scipy.sparse.linalg import svds
from scipy.linalg import svd

import open3d as o3d
# import trimesh

from scipy.spatial import Delaunay
import copy
from matplotlib import cm
import mcgpy
from mcgpy.timeseries import TimeSeriesArray


# from pyarrow import csv
__author__ = 'wonsik.Jung'
__credit__ = 'AMCG'
__all__ = ['RemoveCircuitPulseNoise', 'Gain', 'ByteBuffer', 'Indexes', 'SignalDecimate', 'ReadIndexies',
           'butter_bandpass', 'butter_bandpass', 'Move_position', 'setting_3D_Vertex', 'cart2sph',
           'sph2cart', 'make_3d_axis', 'check_angle', 'get_rotation_matrix', 'rotation_matrix_from_vectors', 'Read_INP',
           'Read_source_amp', 'rotate_view', 'visulize_Egg', 'setting_2D_Vertex', 'Read_2D_source_amp']


def RemoveCircuitPulseNoise(data):
    data_length = len(data)
    data.append(np.median(data).astype(np.int32))
    for i in range(data_length + 1):
        abs_data = np.abs(data)
        max_value = abs_data.max()
        max_index = int(np.where(abs_data == max_value)[0][0])
        indicator = abs_data[max_index + 1] * 100
        if max_value > indicator:
            data[max_index] = data[max_index + 1]
            print(i)
            continue
        else:
            break
    del data[-1]

    return data


def Gain(system_gain, minimum_range, maximum_range):
    if system_gain == 0:  # default
        gain = np.multiply((maximum_range - minimum_range), 0.5 * 1000).astype(np.float32)
    elif system_gain == 2:
        gain_list = [1.140000000000,
                     1.064689748547,
                     1.070348223079,
                     1.050158169168,
                     1.060617048626,
                     1.059086250139,
                     1.126160308754,
                     1.077326284194,
                     1.062401477637,
                     1.152351232444,
                     1.065760732372,
                     1.075829971346,
                     1.055526677748,
                     1.064517859654,
                     1.166788764423,
                     1.079126627173,
                     1.080095985567,
                     1.062509901709,
                     1.065555690346,
                     1.078407694924,
                     1.150345577639,
                     1.077350401531,
                     1.192532578579,
                     1.061794750449,
                     1.061089271223,
                     1.085328405243,
                     1.052599945100,
                     1.428122894033,
                     1.091450871449,
                     1.061175768896,
                     1.410562003331,
                     1.433198172062,
                     1.140000000000,
                     1.301986732547,
                     1.415893602342,
                     1.191230725471,
                     1.059633080335,
                     1.086617814525,
                     1.063616086146,
                     1.138060094660,
                     1.047281700359,
                     1.076386919286,
                     1.151940116141,
                     1.058986024518,
                     1.077617448931,
                     1.140000000000,
                     1.030989030763,
                     1.073524410944,
                     1.090633751181,
                     1.054948264372,
                     1.138199158978,
                     1.071383292277,
                     1.040416960920,
                     1.137045026111,
                     1.048734102818,
                     1.149064905471,
                     1.028813063165,
                     1.078903604167,
                     1.140000000000,
                     1.199994033713,
                     1.079938886773,
                     1.033233807718,
                     1.044074911433,
                     1.072169537215,
                     1.176435594141,
                     1.037858286672,
                     1.148428378168,
                     1.174842066022,
                     1.020914518073,
                     1.038982376505,
                     1.173348158927,
                     1.044333416997,
                     1.173844076390,
                     1.042117099332,
                     1.055573343415,
                     1.030751837315,
                     1.039936752130,
                     1.025335616008,
                     1.060059307272,
                     1.055810015441,
                     1.036717322305,
                     1.024154426560,
                     1.143227568004,
                     1.140000000000,
                     1.072616190496,
                     1.150946714058,
                     1.051182229301,
                     1.049941092044,
                     1.182420127874,
                     1.045531484438,
                     1.026363065545,
                     1.041587132894,
                     1.053819344405,
                     1.071080545675,
                     1.063351898675,
                     1.057459720025,
                     1.064779710382,
                     1.075071719141,
                     1.038966960226,
                     1.044794054628,
                     1.098482332885,
                     1.061290241370,
                     1.066669825734,
                     1.029183020706,
                     1.055185392919,
                     1.054790187550,
                     1.060563509756,
                     1.053508675057,
                     1.054081998637,
                     1.061709397420,
                     1.046475944581,
                     1.026947892671,
                     1.084362963570,
                     1.058068114132,
                     1.055476313546,
                     1.433779965563,
                     1.103192692009,
                     1.418022365520,
                     1.140000000000,
                     1.140000000000,
                     1.120565662306,
                     1.029084691992,
                     1.361926919837,
                     1.039724141294,
                     1.083415637174,
                     1.024977588840,
                     1.076791999544,
                     1.067673594861,
                     1.057560953367,
                     1.025654043117,
                     1.063429059235,
                     1.049615198078,
                     1.168312578411,
                     1.058229875971,
                     1.055080985371,
                     1.058835173974,
                     1.031553853911,
                     1.076070448322,
                     1.046594225134,
                     1.049261554357,
                     1.124285111176,
                     1.183791924336,
                     1.068252949333,
                     1.051679022988,
                     1.037103549443,
                     1.069314059916,
                     1.050266035040,
                     1.044657590135,
                     1.056374100329,
                     1.069658646060,
                     1.027078560117,
                     1.060769935730,
                     1.140000000000,
                     1.140000000000,
                     1.140000000000,
                     1.140000000000,
                     1.140000000000,
                     1.140000000000,
                     1.140000000000,
                     1.140000000000]
        for i in range(channels_number - 1):
            if i == 0:
                gain = np.array(gain_list[i] * 1000000, dtype=np.float32)
            else:
                gain = np.append(gain, gain_list[i] * 1000000)

    elif system_gain == 3:
        gain_list = [1.314415646676,
                     -1.139521315572,
                     -1.161156095399,
                     -1.138594234812,
                     -1.152910591016,
                     -1.144180498722,
                     -1.167850684976,
                     -1.174256750587,
                     -1.171135209544,
                     -1.188198472199,
                     -1.159384966735,
                     -1.186816634865,
                     -1.189088031125,
                     -1.160415778838,
                     -1.167612881852,
                     -1.172338258359,
                     -1.142680802565,
                     -1.161401278128,
                     -1.140634914326,
                     -1.167334552228,
                     -1.149613253317,
                     -1.163753091030,
                     -1.176427010234,
                     -1.196641095214,
                     -1.175096984100,
                     -1.171878190721,
                     -1.198702243888,
                     -1.199416378432,
                     -1.190142329984,
                     -1.179703459870,
                     -1.192893734116,
                     -1.178096925690,
                     -1.142894494732,
                     -1.144662220955,
                     -1.101100017652,
                     -1.164585814441,
                     -1.134294711830,
                     -1.150047404218,
                     -1.200757756690,
                     -1.150956803556,
                     -1.152092961320,
                     -1.211468966197,
                     -1.162318388020,
                     -1.166083266478,
                     -1.169919135388,
                     -1.184487574481,
                     -1.179890735752,
                     -1.241244380666,
                     -1.128299245860,
                     -1.128866308748,
                     -1.125745453379,
                     -1.140238704396,
                     -1.145380829858,
                     -1.147378606864,
                     -1.141008180818,
                     -1.159056513792,
                     -1.151982636796,
                     -1.152855479899,
                     -1.185094411175,
                     -1.159483089406,
                     -1.162791536264,
                     -1.178744039646,
                     -1.165747629920,
                     -1.177771666268,
                     -1.182690586699,
                     -1.253801830758,
                     -1.200224434090,
                     -1.182591964472,
                     -1.190679374430,
                     -1.171493736460,
                     -1.189300076280,
                     -1.178677452144,
                     -1.199821372178,
                     -1.180688492857,
                     -1.172903561695,
                     -1.193767676052,
                     -1.187936200647,
                     -1.194971602916,
                     -1.200665692316,
                     -1.187762287541,
                     -1.160559041705,
                     -1.183803636741,
                     -1.183725349092,
                     -1.160610736566,
                     -1.198013738260,
                     -1.174036103724,
                     -1.200396954508,
                     -1.184666792439,
                     -1.192628070102,
                     -1.183863360496,
                     -1.196667288150,
                     -1.195469674868,
                     -1.184413988755,
                     -1.168998371815,
                     -1.168344247902,
                     -1.185641255590,
                     -1.152116944560,
                     -1.153106434254,
                     -1.115210106628,
                     -1.140000000000,
                     -1.141510394002,
                     -1.148356517140,
                     -1.164290291130,
                     -1.150998795176,
                     -1.170569642247,
                     -1.154314753753,
                     -1.176852873165,
                     -1.164542559894,
                     -1.188805375040,
                     -1.168318269320,
                     -1.180019786326,
                     -1.180052827406,
                     -1.143834248944,
                     -1.148601659913,
                     -1.140185607063,
                     -1.130807828720,
                     -1.147620518252,
                     -1.170586905844,
                     -1.159476934454,
                     -1.136407701395,
                     -1.184261553467,
                     -1.277222064545,
                     -1.181568750923,
                     -1.175547856815,
                     -1.200360353130,
                     -1.162449954141,
                     -1.157571050937,
                     -1.179993190698,
                     -1.221861343715,
                     -1.219744756909,
                     -1.140000000000,
                     -1.190960674359,
                     -1.231009980378,
                     -1.197872049925,
                     -1.183071121343,
                     -1.190111901862,
                     -1.209167666825,
                     -1.199752521400,
                     -1.207436252164,
                     -1.200229494167,
                     -1.184392949574,
                     -1.186650897704,
                     -1.181283374572,
                     -1.201409467875,
                     -1.145220336358,
                     -1.158486344606,
                     -1.149954651365,
                     -1.185454776512,
                     -1.185452100080,
                     -1.174776892114,
                     -1.184630266220,
                     -1.210898000851,
                     -1.140000000000,
                     -1.140000000000,
                     -1.140000000000,
                     -1.140000000000,
                     -1.140000000000,
                     -1.140000000000,
                     -1.140000000000,
                     -1.140000000000]

        for i in range(channels_number - 1):
            if i == 0:
                gain = np.array(gain_list[i] * 1000000, dtype=np.float32)
            else:
                gain = np.append(gain, gain_list[i] * 1000000)
    else:
        raise ValueError('An incorrect value was inputted. Available values are 0, 2, or 3.')

    return gain


def ByteBuffer(data_segment):
    x = data_segment[0]
    y = data_segment[1]
    z = data_segment[2]

    u16 = np.frombuffer(x.tobytes() + y.tobytes(), dtype=np.uint16)[0]
    u32 = np.frombuffer(u16.tobytes() + np.uint16(z).tobytes(), dtype=np.uint32)[0]
    i32 = np.int32(u32)

    if i32 > 8388607:
        i32 = i32 - 16777216

    return i32


def Indexes(decimating_factor, index_info_box):
    for i, index_info in enumerate(index_info_box):
        if i == 0:
            trigger_indexes = np.array(index_info[:, 0])
            response_indexes = np.array(index_info[:, 1])
        else:
            trigger_indexes = np.append(trigger_indexes, index_info[:, 0])
            response_indexes = np.append(response_indexes, index_info[:, 1])

    if decimating_factor == 0 or decimating_factor == 1:  # default
        return trigger_indexes, response_indexes

    elif decimating_factor == 2:
        for j in range(int((time_duration * sampling_rate) / 2)):
            if j == 0:
                trigger_indexes = np.array(
                    trigger_indexes.reshape(int((time_duration * sampling_rate) / 2), 2)[i].max())
                response_indexes = np.array(
                    response_indexes.reshape(int((time_duration * sampling_rate) / 2), 2)[i].max())
            else:
                trigger_indexes = np.append(trigger_indexes,
                                            trigger_indexes.reshape(int((time_duration * sampling_rate) / 2), 2)[
                                                i].max())
                response_indexes = np.append(response_indexes,
                                             response_indexes.reshape(int((time_duration * sampling_rate) / 2), 2)[
                                                 i].max())
        return trigger_indexes, response_indexes

    else:
        raise ValueError('An incorrect value was inputted. Available values are 0, 1, or 2.')


def SignalDecimate(signal, sampling_rate, decimating_factor):
    signal_out = signal.reshape(int(sampling_rate / decimating_factor), decimating_factor)[:, 0]
    return signal_out


def ReadIndexies(path, start, end, decimating_factor=1):
    KDF_data_size = os.path.getsize(path)  # file size in bytes
    with open(path, 'br') as f:
        code = f.read(1)
        BIOSEMI = f.read(7).decode('ascii').strip()
        subject_info = f.read(80).decode('ascii').strip()  # Require to decode Patient name
        recording_info = f.read(80).decode('ascii').strip()
        date_info = f.read(8).decode('ascii').strip()  # DD.MM.YY
        for i, value in enumerate(date_info.split('.')[::-1]):
            if i == 0:
                YY = int('20' + value)
            elif i == 1:
                MM = int(value)
            elif i == 2:
                DD = int(value)
        time_info = f.read(8).decode('ascii').strip()  # hh.mm.ss
        for i, value in enumerate(time_info.split('.')):
            if i == 0:
                hh = int(value)
            elif i == 1:
                mm = int(value)
            elif i == 2:
                ss = int(value)
        datetime_info = datetime.datetime(YY, MM, DD, hh, mm, ss)  # datetime format
        header_byte = int(f.read(8).decode('ascii').strip())
        data_format = f.read(44).decode('ascii').strip()  # 24 bit
        data_records = int(f.read(8).decode('ascii').strip())  # seconds / -1 means unknown / interger
        duration = int(f.read(8).decode('ascii').strip())
        channels_number = int(f.read(4).decode('ascii').strip())
        channel_labels = f.read(16 * (channels_number)).decode('ascii')
        channel_labels = [channel_labels[i * 16:(i + 1) * 16].strip() for i in range(channels_number - 1)]
        coil_types = f.read(40 * (channels_number)).decode('ascii')
        units = f.read(8 * (channels_number)).decode('ascii')
        minimum_range = f.read(8 * (channels_number)).decode('ascii')
        digital_minimum = f.read(8 * (channels_number)).decode('ascii')
        digital_maxmum = f.read(8 * (channels_number)).decode('ascii')
        prefiltering = f.read(80).decode('ascii').strip()
        sampling_rate = int(f.read(8).decode('ascii'))

        recording_time = int((KDF_data_size - header_byte) / (
                    channels_number * sampling_rate * 3))  # mesearment time have to be equal to data_records time

        if start < 0:
            start = 0
        elif start > 0:
            datetime_info = datetime_info + datetime.timedelta(seconds=start)
        start = min(recording_time - 1, start)
        offset = start * (sampling_rate * 3) * (channels_number)
        count = 3 * sampling_rate
        time_duration = min(recording_time, recording_time - start)

        bdata = np.fromfile(f, offset=offset, dtype=np.uint8)
        datasets = bdata.reshape(int(bdata.shape[0] / count), count)
        index_info_box = list()
        for time, dataset in enumerate(np.array_split(datasets, time_duration)):
            index_info = dataset[-1].reshape(sampling_rate, 3)
            if time < end - start:
                index_info_box.append(index_info)
            else:
                break
        f.close()

    metainfo = np.dtype(np.float32, metadata={'BIOSEMI': BIOSEMI,
                                              'subject_info': subject_info,
                                              't0': datetime_info,
                                              'channels number': channels_number,
                                              'duration': end - start,
                                              'sampling rate': sampling_rate,
                                              'prefiltering': prefiltering
                                              })
    trigger_train, response_train = Indexes(decimating_factor, index_info_box)
    trigger_train = trigger_train.astype(metainfo)
    response_train = response_train.astype(metainfo)

    return trigger_train, response_train


def ReadTimeseries(path, start, end, decimating_factor=1, gainfactor=1):
    KDF_data_size = os.path.getsize(path)  # file size in bytes
    with open(path, 'br') as f:
        code = f.read(1)
        BIOSEMI = f.read(7).decode('ascii').strip()

        subject_info = f.read(80).decode('ascii').strip()  # Require to decode Patient name
        recording_info = f.read(80).decode('ascii').strip()
        try:
            system_gain = int(recording_info.split(' ')[4])
        except:
            system_gain = 3

        date_info = f.read(8).decode('ascii').strip()  # DD.MM.YY
        for i, value in enumerate(date_info.split('.')[::-1]):
            if i == 0:
                YY = int('20' + value)
            elif i == 1:
                MM = int(value)
            elif i == 2:
                DD = int(value)
        time_info = f.read(8).decode('ascii').strip()  # hh.mm.ss
        for i, value in enumerate(time_info.split('.')):
            if i == 0:
                hh = int(value)
            elif i == 1:
                mm = int(value)
            elif i == 2:
                ss = int(value)

        datetime_info = datetime.datetime(YY, MM, DD, hh, mm, ss)  # datetime format

        header_byte = int(f.read(8).decode('ascii').strip())
        data_format = f.read(44).decode('ascii').strip()  # 24 bit
        data_records = int(f.read(8).decode('ascii').strip())  # seconds / -1 means unknown / interger
        duration = int(f.read(8).decode('ascii').strip())
        channels_number = int(f.read(4).decode('ascii').strip())

        channel_labels = f.read(16 * (channels_number)).decode('ascii')
        channel_labels = [channel_labels[i * 16:(i + 1) * 16].strip() for i in range(channels_number - 1)]

        coil_types = f.read(40 * (channels_number)).decode('ascii')
        coil_types = [coil_types[i * 40:(i + 1) * 40].strip() for i in range(channels_number - 1)]

        units = f.read(8 * (channels_number)).decode('ascii')
        units = [units[i * 8:(i + 1) * 8].strip() for i in range(channels_number - 1)]

        minimum_range = f.read(8 * (channels_number)).decode('ascii')
        minimum_range = np.array([float(minimum_range[i * 8:(i + 1) * 8].strip()) for i in range(channels_number - 1)])

        maximum_range = f.read(8 * (channels_number)).decode('ascii')
        maximum_range = np.array([float(maximum_range[i * 8:(i + 1) * 8].strip()) for i in range(channels_number - 1)])

        digital_minimum = f.read(8 * (channels_number)).decode('ascii')
        digital_minimum = np.array(
            [float(digital_minimum[i * 8:(i + 1) * 8].strip()) for i in range(channels_number - 1)])

        digital_maxmum = f.read(8 * (channels_number)).decode('ascii')
        digital_maxmum = np.array(
            [float(digital_maxmum[i * 8:(i + 1) * 8].strip()) for i in range(channels_number - 1)])

        prefiltering = f.read(80).decode('ascii').strip()
        sampling_rate = int(f.read(8).decode('ascii'))

        recording_time = int((KDF_data_size - header_byte) / (
                    channels_number * sampling_rate * 3))  # mesearment time have to be equal to data_records time

        gain = Gain(system_gain, minimum_range, maximum_range)

        if start < 0:
            start = 0
        elif start > 0:
            datetime_info = datetime_info + datetime.timedelta(seconds=start)
        start = min(recording_time - 1, start)
        offset = start * (sampling_rate * 3) * (channels_number)
        count = 3 * sampling_rate
        time_duration = min(recording_time, recording_time - start)

        bulkdataset = list()

        bdata = np.fromfile(f, offset=offset, dtype=np.uint8)
        datasets = bdata.reshape(int(bdata.shape[0] / count), count)
        index_info_box = list()
        for time, dataset in enumerate(np.array_split(datasets, time_duration)):
            ch_dataset = np.delete(dataset, -1, axis=0)
            if time < end - start:
                for i, ch_data in enumerate(ch_dataset):
                    data = ch_data.reshape(sampling_rate, 3)
                    bin = list()
                    for row in data:
                        bin.append(ByteBuffer(row))

                    RCPN = np.array(RemoveCircuitPulseNoise(bin)).astype(np.int32)
                    decimated_signal = SignalDecimate(RCPN, sampling_rate, decimating_factor)

                    if i == 0:
                        out_data = decimated_signal
                    else:
                        out_data = np.vstack((out_data, decimated_signal))
                bulkdataset.append(out_data)
            else:
                break
        f.close()

    package = dict()
    if gain.shape[0] == 0:
        for i, channel_label in enumerate(channel_labels):
            metainfo = np.dtype(np.int32, metadata={'BIOSEMI': BIOSEMI,
                                                    'subject_info': subject_info,
                                                    't0': datetime_info,
                                                    'duration': end - start,
                                                    'channel label': channel_label,
                                                    'coil type': coil_types[i],
                                                    'unit': units[i],
                                                    'sampling rate': sampling_rate,
                                                    'data format': data_format,
                                                    'prefiltering': prefiltering
                                                    })
            for time, timeseries in enumerate(bulkdataset):
                if time == 0:
                    package[channel_label] = timeseries[i].astype(metainfo)
                else:
                    package[channel_label] = np.append(package[channel_label], timeseries[i].astype(metainfo))

    else:
        gain = np.divide(gain * gainfactor, 838860.8)
        for i, channel_label in enumerate(channel_labels):
            metainfo = np.dtype(np.float32, metadata={'BIOSEMI': BIOSEMI,
                                                      'subject_info': subject_info,
                                                      't0': datetime_info,
                                                      'duration': end - start,
                                                      'channel label': channel_label,
                                                      'coil type': coil_types[i],
                                                      'unit': units[i],
                                                      'sampling rate': sampling_rate,
                                                      'data format': data_format,
                                                      'prefiltering': prefiltering
                                                      })
            for time, timeseries in enumerate(bulkdataset):
                if time == 0:
                    package[channel_label] = (timeseries[i] * gain[i]).astype(metainfo)
                else:
                    package[channel_label] = np.append(package[channel_label],
                                                       (timeseries[i] * gain[i]).astype(metainfo))

    return package


## This Function is related to angle,move in 3D or 2D position Vector
## derived from this Function Rotation, Move(All Position SHIFT) Function, projection Matrix

def Move_position(node_position):
    '''shift Vertexs to Center_position
    also Center or mean to be (0,0,0)

    Parameters
    ----------
    node_position : "list","numpy",etc
        node_position is [N*A] Matrix This Matrix is position information All Node position
        N:node number, A:node position Vector



    Return : "Move_position (numpy)"
    ------
        1) Move_position: all Node position Moved to Center,Center of  Node position will be (0,0,...0,0)



    Examples
    --------
    >>> from Bull_eye import Move_position

    >>> Move_position=Move_position(INP_Data)'''

    # Center_position=Center_Position(node_position)
    Move_position = np.array(node_position, dtype=float)

    for i in range(Move_position.shape[1]):
        Move_position[:, i] = Move_position[:, i] - np.mean(Move_position[:, i])

    return Move_position


def setting_3D_Vertex(heart):
    ''' This Function is related to angle,move in 3D or 2D position Vector
    derived from this Function Rotation, Move(All Position SHIFT) Function, projection Matrix

    Parameters
    ----------
    heart : "str"
        heart is path of INP(Contain Heart_Node_position_Information) file


    Return : "Moved_pos"
    ------
        1) Moved_pos : Moved_position is [N*3]

    Examples
    --------
    >>> from Bull_eye import setting_3D_Vertex


    >>> Moved_pos=setting_3D_Vertex(heart_path)'''

    try:
        vertex, elements, Vertexs, Element = Read_INP(heart)
        Vertexs = np.array(Vertexs, dtype=float)
        Moved_pos = Move_position(Vertexs)
    except:
        print('check_heart_inp_path')
    return Moved_pos


def cart2sph(X):
    ''' Get From XYZ Coordinate to Sphere Coordinate (r,elev,za)

    Parameters
    ----------
    X : 'list',"numpy"
        X is list or numpy, is Vertor [x,y,z]

    Return : "r,elev,az"
    ------
        1) r: r is the radial distance (distance to origin)
        2) elev: elev is the polar Angle (angle woth respect to polar axis)
        3) az: az is the azimuthal (angle of rotation from the initial meridian plane)

    Examples
    --------
    >>> from Bull_eye import cart2sph

    >>>cart2sph([1,1,1])
    ''''

    r = np.linalg.norm(X)
    elev = m.acos(X[2] / r)
    az = m.atan2(X[1], X[0])  # phi
    return r, elev, az


def cart2sph_2D(X):
    ''' Get From XYZ Coordinate to Sphere Coordinate (r,elev,za)
    Extract r,az for plotting 2D plot

    Parameters
    ----------
    X : 'list',"numpy"
        X is list or numpy, is Vertor [x,y,z]



    Return : "r,az"
    ------
        1) r: r is the radial distance (distance to origin)
        2) az: az is the azimuthal (angle of rotation from the initial meridian plane)

    Examples
    --------
    >>> from Bull_eye import cart2sph_2D

    >>>cart2sph_2D([1,1,1])
    ''''

    r = np.sqrt(x ** 2 + y ** 2)
    az = m.atan2(X[1], X[0])  # phi
    return r, az


def sph2cart(X):
    ''' Get From Sphere Coordinate to XYZ, Coordinate

    Parameters
    ----------
    X : "list","numpy"
        X is Sphere Coordinate R, Theta, Pi



    Return : "x,y,z" "numpy"
    ------
        1) "x,y,z" : x,y,z is XYZ coordinate

    Examples
    --------
    >>> import Bull_eye import sph2cart
    '''

    r, theta, p = X
    x = r * m.sin(theta) * m.cos(p)
    y = r * m.sin(theta) * m.sin(p)
    z = r * m.cos(theta)
    return x, y, z


def make_3d_axis(X):
    ''' Get Basis From some X_1,Y_1,Z_1, make 3D Basis Axis
    #Use dot product Caculate Between Two_vector

    Parameters
    ----------
    X : "list","numpy",etc
        X is Data type, or Timeseriesarray Data



    Return : "[z1,z2,z3]" ,"list","basis"
    ------
        1) [z1,z2,z3] is Basis of

    Examples
    --------
    >>> from Bull_eye import make_3d_axis

    >>> output=make_3d_axis([1,1,1])
    '''

    amplitude = np.linalg.norm(X)
    unit_vector = np.array(X) / np.linalg.norm(X)
    x = unit_vector[0]
    y = unit_vector[1]
    z = unit_vector[2]

    sph = cart2sph(unit_vector)
    z1 = sph2cart(np.array(sph) + np.array([0, m.pi / 2, 0]))
    z3 = unit_vector
    z2 = np.cross(z3, z1)

    output = np.array([z1, z2, z3])

    return output


def check_angle(x1, x2):
    '''
Get Angle between x1 and x2

Parameters
----------
x1 : "list","numpy"
    x1 is First Vector

x2 : "list","numpy"
    x2 is Second Vector



Return : "theta_list"
------
    1) theta_list : theta_list is Angle Between x1,x2

Examples
--------
>>> from Bull_eye import check_angle
>>> x1=[1,1,1],
>>> x2=[2,1,2]

>>> gap_angle=check_angle(x1,x2)'''


theta_list = []
for i in range(x1.shape[-1]):
    v_i = np.inner(x1[i], x2[i]) / (np.inner(x1[i], x1[i]) * np.inner(x2[i], x2[i]))
    theta = np.arccos(v_i)
    theta_list.append(theta)
return theta_list


def get_rotation_matrix(axis, theta):
    '''Find the rotation matrix associated with counterclockwise rotation
    about the given axis by theta radians.
    Credit: http://stackoverflow.com/users/190597/unutbu

    Args:
        axis (list): rotation axis of the form [x, y, z]
        theta (float): rotational angle in radians

    Returns:
        array. Rotation matrix.


    Get R_Peak Time and Value Informataion

    Parameters
    ----------
    axis : "list","numpy"
        axis will is Standard in 3d Coordinate
        rotation axis of the form [x, y, z]


    theta : "float"
        rotational angle(radius) from axis


    Return : "rotataion Matrix"
    ------
        1) [[r_1][r_2][r_3]] :
           return the values 'list'


    Examples
    --------
    >>> from Bull_eye import get_rotation_matrix
    >>> axis=[[1,0,0],[0,1,0],[0,0,1]]
    >>> theta=[60]

    >>> rotated_result=get_rotation_matrix(axis,theta)'''

    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / m.sqrt(np.dot(axis, axis))
    a = m.cos(theta / 2.0)
    b, c, d = -axis * m.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotation_matrix_from_vectors(vec1, vec2):
    ''' Find the rotation matrix that Angle vec1 and vec2
    :param vec1: 3d "source" vector
    :param vec2: 3d "destination" vector
    :return mat: transform matrix (3x3) which when applied to vec1, aligns it with vec2.

    Parameters
    ----------
    vec1 : "list","numpy"
        vec1 is 3d Verctor and it is source vector

    vec2 : "list" or "str"
        vec2 is 3d Verctor and it is destination vector    

    Return : "Rotation_matrix"
    ------
        1) Rotation_Matrix vec1 and vec2
        vec 1 Rotated to vec 2

    Examples
    --------
    >>> from Bull_eye import rotation_matrix_from_vectors
    >>> x1=[1,1,1]
    >>> x2=[2,1,2]

    >>> result=rotation_matrix_from_vectors(x1,x2)''''

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


# This function related to Read Color value
# derived from INP File, KDF file data with this function
# Get Source_amplitude, transfer rgb2gray,gray2rgba(Labview)

def Read_INP(path):
    ''' Get INP File information from path

     information is Contained
     (Number of Vertex, Number of Mesh, Array of Vertex, Array of Mesh,Make Vertex position and Triangle Mesh information)


    Parameters
    ----------
    path : "str"
        path is path of INP file


    Return : "vertex,elements,Vertexs,Triangle_info"

    ------
        1) vertex : "int"
        The number of Vertexs(N_1)

        2) elements: "int"
        The number of Triangle_info(N_2)

        3) Vertexs: "numpy"
        Node [x,y,z] Position Vectors [N_1*3]

        4) Trianlge_info: "numpy"
        deluary Triangle Combination information [N_2*3]


    Examples
    --------
    >>> from Bull_eye import Read_INP

    >>> N_vertex,N_elements,Vertexs,element=Read_INP(INP_path)'''

    line = []
    try:
        with open(path, 'r') as f:

            a = f.read(64)
            vertex = f.read(3).strip()
            ee = f.read(11).strip()
            elements = ee[:3]

            for i in range(int(vertex) + int(elements)):
                aa = f.readline()
                aa = aa.strip("\n")
                ab = aa.split(' ')
                line.append(ab)

            Vertexs = np.array(line[:int(vertex)])[:, 1:]

            Element = line[int(vertex):]
            Element = np.array(Element)[:, 3:]
            Triangle_info = [x[::-1] for x in np.array(Element, dtype=float)]

            return vertex, elements, np.array(Vertexs, dtype=float), Triangle_info

    except:
        print('check file path,file type, or import numpy as np')


def Read_source_amp(kdf_dataes, lead_file, timecursor, Eigenvalue=int(11), normalization=False):
    ## READ epoch dataset
    '''Get Source Amplitude
    Data Structure
     EX: kdf_dataes=ReadTimeseries(kdf_file, start, end, decimating_factor=1, gainfactor=1)
     time cursor= set Read time cursor (in your signal data time)'''
    '''


Parameters
----------
kdf_dataes : "numpy"
    kdf_dataes is Timeseriesarray Data

lead_file : "str"
    lead_file is data path of lead_field


timecursor : "float" or "int"
    Set you'r TimeSeriesArray Data time index
    Caculate from at this time

Eigenvalue : "float" ot "int"
    Set Eigenvalue, When You Use MNE Algorithms
    Caculate from this Eigenvalue

normalization: "str" (Default=False)
    Set your Amplitude Normalize(./max(Amplitude))


Return : "source_amp"
------
    1) source_amp: Value(Amplitude) in Time Index in kdf_dataset


Examples
--------
>>> from Bull_eye import Read_source_amp

>>> Amplitude=Read_source_amp(kdf_data,'lead_field\aaa.csv',310)'''


with open(lead_file, 'r') as f:
    for i, row in enumerate(f.read().split('\n')):
        try:
            if i == 1:
                leadfield = np.array([np.float64(value) for value in row.split(',')])
            elif i > 1:
                leadfield = np.vstack((leadfield, np.array([np.float64(value) for value in row.split(',')])))
        except ValueError:
            pass

# '''1) Make lead field Structure, From your lead field path extract all channel'''


for i, ch in enumerate(kdf_dataes.values()):
    if i == 0:
        dataset = ch
        reduced_leadfield = leadfield[i]
    else:
        if np.sum(ch) != 0:
            dataset = np.vstack((dataset, ch))
            reduced_leadfield = np.vstack((reduced_leadfield, leadfield[i]))

# '''2)Make kdf data and reduced lead field Structure (Based on Activated Channel), From your kdf file'''


##########################
## make diagonal norm matrix
diagonal_norm_matrix = np.diag(np.sqrt(1 / np.linalg.norm(reduced_leadfield, axis=0)))

## calculate SVD
special_matrix = np.dot(reduced_leadfield, diagonal_norm_matrix)
u, s, vh = np.linalg.svd(special_matrix, full_matrices=True)

## calculate inverse matrix
eigenvalues = 11
fractional_index = np.where(s[::-1] > np.multiply(np.sum(s), 0.01))[0][0]
eigenvalues = s.shape[0] - np.int16(fractional_index) - 6

b = np.dot(np.diag(1 / s[:eigenvalues]), u[:, :eigenvalues].T)
a = np.dot(vh.T[:, :eigenvalues], b)

inverse_leadfield = np.dot(diagonal_norm_matrix, a)

########################


#     U, Sigma, Vt = svd(reduced_leadfield, full_matrices=False)

#     Ut,V=np.transpose(U),np.transpose(Vt)
#     inverse_Sigma= np.diag([1/x for x in Sigma if x >= Sigma[Eigenvalue]])

inverse_leadfield = np.dot(np.dot(V, inverse_Sigma), Ut)

# '''3) Make Inverse Lead field, Eigenvalue is 11
# Make inverse leadfield use scipy.linalg.SVD and Pseudoinverse'''

epoch_dataset = dataset[:, int(timecursor)]
source = np.dot(inverse_leadfield, epoch_dataset).reshape(298, 3)
source_amp = np.array([np.linalg.norm(vector) for vector in source])
source_amp_normalized = (source_amp) / (np.max(source_amp))

# '''4) Calculate Normalized Source Amplitude, from inverse lead field and dataset[:,time]'''

if normalization == False:
    return source_amp
else:
    return source_amp_normalized


def rgb2gray(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

    # '''Gray Color transfer, fromm Rgb to Gray'''


## Read Leadfield file
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as mani
import math as m
import copy


def rotate_view(vis):
    '''The Function for auto Rotation,
    This function is Call-Back Function


    Parameters
    ----------
    vis :



    Return : False
    ------
        1) "rotate(-4,0)"'''

    ctr = vis.get_view_control()
    ctr.rotate(-4.0, 0.0)
    return False


def visulize_Egg(normalized_3d_norm, axis=None, heart=heart):
    '''Visualizing Egg from amplitude and heart Model

    Parameters
    ----------
    normalized_3d_norm : "numpy"
        Norm Based on 3d amplitude

    axis : "int" or "str"
        Set Visualizeing Options

    heart : "str"
        Set heart_ model INP path


    Return : Open 3d visualizing Window
    ------

    Examples
    --------
    >>> from Bull_eye import visulize_Egg

    >>>visulize_Egg(normalized_3d_norm,heart=inp_path)'''

    vertex, elements, Vertexs, Element = Read_INP(heart)
    Vertexs = np.array(Vertexs, dtype=float)

    Moved_pos = Move_position(Vertexs)
    triangle = np.asarray(Element, dtype=int)

    # '''Set Color option, We Use 'jet' u can Other Color Option in Cmap function '''
    # normalized_3d_norm=Read_source_amp(kdf_dataes,lead_file,timecursor,Eigenvalue=int(11))
    normalized_3d_norm = normalized_3d_norm
    # normalized_3d_norm=np.array(normalized_3d_norm)/np.max(normalized_3d_norm)

    viridis = cm.get_cmap('jet')

    colors = list()
    for amp in normalized_3d_norm:
        colors.append(list(viridis(amp)[:-1]))

        ###
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(Moved_pos)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    stl_mesh = o3d.geometry.TriangleMesh()

    stl_mesh.vertices = o3d.utility.Vector3dVector(1.2 * on_to_on_point)
    stl_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))
    stl_mesh.triangles = o3d.utility.Vector3iVector(triangle)
    stl_mesh.get_center()

    # '''Set Mesh option, Vertex & Triange mesh and Color '''
    frontal = o3d.geometry.TriangleMesh()

    frontal.vertices = o3d.utility.Vector3dVector(Moved_pos)
    frontal.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))
    frontal.triangles = o3d.utility.Vector3iVector(triangle)
    frontal.get_center()

    axis_coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
    axis_coordinate.scale(200, center=axis_coordinate.get_center())
    moved_axis_coordinate = copy.deepcopy(axis_coordinate).translate((0, 200, 200), relative=False)

    ###
    standard_model_Lateral = copy.deepcopy(stl_mesh)
    standard_model_Lateral.rotate(
        o3d.geometry.get_rotation_matrix_from_xyz([-1 / 2 * m.pi, -1 / 2 * m.pi, -1 / 2 * m.pi]),
        center=standard_model_Lateral.get_center())

    standard_model_inferior = copy.deepcopy(standard_model_Lateral)
    standard_model_inferior = standard_model_inferior.rotate(
        o3d.geometry.get_rotation_matrix_from_zxy([-1 / 2 * m.pi, -1 / 2 * m.pi, -1 / 2 * m.pi]),
        center=standard_model_inferior.get_center())

    # '''Set frontal,Laternal, Inferior, Shallow Copy frontal'''

    Lateral = copy.deepcopy(frontal)
    Lateral.rotate(o3d.geometry.get_rotation_matrix_from_xyz([-1 / 2 * m.pi, -1 / 2 * m.pi, -1 / 2 * m.pi]),
                   center=frontal.get_center())

    inferior = copy.deepcopy(Lateral)
    inferior = inferior.rotate(o3d.geometry.get_rotation_matrix_from_zxy([-1 / 2 * m.pi, -1 / 2 * m.pi, -1 / 2 * m.pi]),
                               center=frontal.get_center())

    # Make coordinate frame

    # '''Visulalization Frontal,Lateral,inferor
    # If you want One image With rotation
    # put axis_list='f'or'l'or'i', '''
    axis_list = [['frontal', 'f', 1], ['lateral', 'l', 2], ['inferior', 'i', 3]]

    if axis in axis_list[0]:
        o3d.visualization.draw_geometries_with_animation_callback([Lateral, standard_model_Lateral], rotate_view,
                                                                  width=800, height=600)
    elif axis in axis_list[1]:
        o3d.visualization.draw_geometries_with_animation_callback([Lateral, standard_model_Lateral], rotate_view,
                                                                  width=800, height=600)
    elif axis in axis_list[2]:
        o3d.visualization.draw_geometries_with_animation_callback([inferior, standard_model_inferior], rotate_view,
                                                                  width=800, height=600)

    else:

        o3d.visualization.draw_geometries([frontal], width=800, height=600, mesh_show_wireframe=True)
        o3d.visualization.draw_geometries([Lateral], width=800, height=600, mesh_show_wireframe=True)
        o3d.visualization.draw_geometries([inferior], width=800, height=600, mesh_show_wireframe=True)

        pass


# by 3D Spheroid Model--Bull's eye positiong of 2D Polar Map model(LAB_VIEW)

def setting_2D_Vertex(heart, rotate=False, append=True):
    ''' This Function is related to angle,move in 3D or 2D position Vector
     derived from this Function Rotation, Move(All Position SHIFT) Function, projection Matrix


    Parameters
    ----------
    heart : "str"
        heart is path of heart INP file

    rotate : "str" (Default = False)
        Setting rotate

    append : "str" (Default = True)
        Set Append option
        plot 2D image, Smoothly

    Return : "xy_node"
    ------
        1) xy_node : 3D Node Project 2D

           '''

    try:
        vertex, elements, Vertexs, Element = Read_INP(heart)
        origin_basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        #         '''1) Set (x,y,z) coordinate basis to apex direction(x1,y1,z1)
        #            (Transfered (x1,y1,z1) to (1,1,1))'''

        direction_basis = make_3d_axis(np.array(Vertexs[0] - Vertexs[1], dtype=float))
        rotated = np.dot(direction_basis, Vertexs.T).T
        rotated_move = rotated - rotated[1]

        # '''2) Make xy_node, Projected to (x1,y1,0) plane '''
        handing_z = np.array([(100 - 0.75 * x[2]) for x in rotated_move])
        # 100 is max(z)
        # To Uniformly disturibute, Technique Idea applied(Multiply 0.75)

        theta_node = np.array([m.atan2(x[1], x[0]) for x in rotated_move])

        # except:
        #     print('check_heart_inp_path')

        x_node = np.array([m.sin(x) for x in theta_node])
        y_node = np.array([m.cos(x) for x in theta_node])

        # except:
        #     print('check_heart_inp_path')

        xy_node = np.array([handing_z * x_node,
                            handing_z * y_node,
                            np.zeros(len(theta_node))]).T

        # '''3) Make 48 append_node(x2,y2,0), to fill empthy edge Area '''
        theta_append_node = (np.array(range(0, 48, 1)) / 48) * 2 * m.pi + m.pi / 2
        append_node = np.array([[100 * m.sin(x1), 100 * m.cos(x1), 0] for x1 in theta_append_node])

        if append != True:
            return xy_node
        elif rotate == True:
            return rotated
        else:
            return np.vstack([xy_node, append_node])
    except:
        print('check_heart_inp_path')


# SO WE Extract after MNE result to EXCEL, for Reproducing
# Our Labview's Normalization only apply is value/max(value)
# after Reproducing apply MinMax Scaler

path_source_vector = "/home/wonsik.jung/Bulleye_3D/testcode_dataes/Activity_Data_file/620data_try2.xls"
path_source = "/home/wonsik.jung/Bulleye_3D/testcode_dataes/Activity_Data_file/620data_try.xls"

with open(path_source_vector, 'r') as f:
    source_vector = f.read().split('\t')
if source_vector[-1] == '\n':
    source_vector[-1] = source_vector[-1][:-1]

######################################################################

source_vector = np.array(source_vector).astype(np.float64).reshape(-1, 3)

normalized_2d_norm = norm_2d / max(norm_2d)

'''def Read_source_amp(kdf_dataes,lead_file,timecursor,Eigenvalue=int(11),normalization=True):'''


def Read_2D_source_amp(source_amp, append=False):
    '''
    Get Amplitude of 2D Node

    Parameters
    ----------
    source_amp : "list","numpy",etc
        3d source_amplitude

    append : "str" (Default=False)
        Set append Option


    Return : "normalized_2d_norm"
    ------
        1) normalized_2d_norm: Amplitude of 2d'''


try:
    norm_2d = np.array([np.linalg.norm(x) for x in source_vector])
    normalized_2d_norm = norm_2d / max(norm_2d)
    # norm_2d=source_amp
except:
    print('''Coult't Read_source_amp, pls check file path kdf_dataes & lead file''')
try:
    '''1) For append Node Color, Define each Area Node'''
    Node_of_0Area = [9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 33, 34, 35, 41, 59, 60, 69, 70, 71, 72, 73, 79, 154]
    Node_of_1Area = [39, 40, 67, 68, 94, 96, 97, 98, 99, 100, 128, 149, 150, 151, 152, 153, 157, 218, 219, 220, 251,
                     252, 253, 256, 257]
    Node_of_2Area = [38, 93, 95, 121, 122, 206, 207, 208, 209, 215, 216, 217, 230, 231, 232, 254, 258, 259, 260, 261,
                     263, 271, 282]
    Node_of_3Area = [24, 37, 90, 91, 92, 119, 120, 125, 126, 127, 162, 163, 164, 204, 205, 212, 213, 214, 227, 229, 269,
                     270, 281, 297]
    Node_of_4Area = [66, 117, 118, 123, 124, 159, 160, 161, 202, 203, 210, 211, 223, 224, 225, 226, 235, 247, 248, 276,
                     277, 286, 294, 295]
    Node_of_5Area = [1, 36, 61, 62, 63, 64, 65, 158, 221, 222, 233, 234, 236, 237, 245, 246, 273, 274, 278, 283, 290,
                     291, 292, 293]
    Node_of_6Area = [8, 17, 32, 42, 43, 44, 58, 77, 78, 155, 156]
    Node_of_7Area = [184, 185, 249, 255, 264, 265, 266, 267, 268, 279, 280]
    Node_of_8Area = [169, 174, 175, 190, 191, 192, 193, 262]
    Node_of_9Area = [139, 177, 194, 195, 196, 197, 228, 238, 239, 272, 287, 296]
    Node_of_10Area = [198, 199, 200, 240, 241, 242, 243, 244, 284, 285]
    Node_of_11Area = [116, 146, 147, 148, 181, 182, 183, 275, 288, 289]
    Node_of_12Area = [5, 6, 7, 16, 29, 30, 31, 45, 46, 47, 57, 74, 75, 76, 89, 186, 187]
    Node_of_13Area = [103, 130, 131, 132, 136, 137, 167, 168, 170, 171, 172, 173, 188, 189, 250]
    Node_of_14Area = [104, 105, 108, 109, 110, 111, 112, 133, 134, 135, 138, 140, 141, 176, 178, 179]
    Node_of_15Area = [56, 86, 87, 88, 106, 113, 114, 115, 142, 143, 144, 145, 165, 180, 201]
    Node_of_16Area = [0, 2, 3, 4, 15, 22, 23, 25, 26, 27, 28, 48, 49, 50, 51, 52, 53, 54, 55, 80, 81, 82, 83, 84, 85,
                      101, 102, 107, 129, 166]

    Node_index = [Node_of_0Area, Node_of_1Area, Node_of_2Area, Node_of_3Area, Node_of_4Area, Node_of_5Area,
                  Node_of_6Area, Node_of_7Area, Node_of_8Area,
                  Node_of_9Area, Node_of_10Area, Node_of_11Area, Node_of_12Area, Node_of_13Area, Node_of_14Area,
                  Node_of_15Area, Node_of_16Area]

    '''2) Caculate Mean Amplitude of all 17 Areas, Pick Mean Amplitude of Edge 6 Areas'''
    Area_Mean_Amplitude = []
    for i, index_number in enumerate(Node_index):
        amplitude_sum = 0
        for index in index_number:
            amplitude_sum += amplitude_norm[index]
        Area_Mean_Amplitude.append(amplitude_sum / len(index_number))

    Deformated_Normalization_Amplitude = Area_Mean_Amplitude / (np.mean(amplitude_norm))
    Amplitude_of_6Areas = Deformated_Normalization_Amplitude[:6]

    '''3) Caculate Mean Amplitude of all 17 Areas, Pick Mean Amplitude of Edge 6 Areas'''
    for k in range(3):
        Lower_Activity = np.hstack((np.array(Amplitude_of_6Areas[-1]), Amplitude_of_6Areas[:-1]))
        Lower_result = 0.5 * (Amplitude_of_6Areas + Lower_Activity)
        result = list(zip(Amplitude_of_6Areas, Lower_result))
        Amplitude_of_append_Node = []
        for i, value in enumerate(result):
            for ii, v2 in enumerate(value):
                Amplitude_of_append_Node.append(v2)

    appended_2d_norm = np.array(list(norm_2d) + Amplitude_of_append_Node)
    normalized_2d_appended_norm = appended_2d_norm / max(appended_2d_norm)
except:
    pass
if append == False:
    return normalized_2d_norm
else:
    return normalized_2d_appended_norm

