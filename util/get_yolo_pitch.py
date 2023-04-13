import numpy as np
import soundfile as sf
import librosa
#输入为参数配置、经过后处理的结果onset\offset\像素点位置、音频路径
def get_yolo_note_pitch(cfg, yolo_res, filepath):
    audio, sr = sf.read(filepath, dtype='float32')
       
    spec = librosa.cqt(audio, sr=sr, hop_length=512, fmin=27.5, n_bins=cfg["pitch_config"]["pitch_n_bins"], bins_per_octave=cfg["pitch_config"]["pitch_bins_per_octave"])
    spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max)


    ttime = cfg["cqt"]['hop'] / cfg["cqt"]['sr']
    yolo_res = np.array(yolo_res)
    yolo_res[:, :2] = yolo_res[:, :2] / ttime
    yolo_pitch = []
    pitch_list = []
    del_list = []

    for i in range(len(yolo_res)):
        note_res = yolo_res[i]
        onset = int(note_res[0])
        offset = int(note_res[1])
        distance_to_bottom = note_res[2]

        y0 = distance_to_bottom / cfg["pitch_config"]['yolox_pixels'] * cfg["pitch_config"]['pitch_n_bins']
        midi_index = int(round(y0))

        energy = spec[max(0, midi_index - cfg["pitch_config"]['First_search_range']): min(midi_index +
                      cfg["pitch_config"]['First_search_range'], spec.shape[0]), onset: offset]

        total_locs = np.argmax(energy, axis=0)
        max_energy_locs = total_locs.mean()
        final_indexs = max_energy_locs - cfg["pitch_config"]['First_search_range'] + midi_index
        total_indexs = total_locs - cfg["pitch_config"]['First_search_range'] + midi_index

        max_energy_locs = np.argmax(
            energy, axis=0) - cfg["pitch_config"]['First_search_range'] + midi_index
        final_midi_notes = librosa.hz_to_midi(
            cfg['cqt']["fmin"] * pow(2, final_indexs / cfg["pitch_config"]['pitch_bins_per_octave']))
        total_midi_notes = librosa.hz_to_midi(
            cfg['cqt']["fmin"] * pow(2, total_indexs / cfg["pitch_config"]['pitch_bins_per_octave']))

        yolo_pitch.append(final_midi_notes)
        pitch_list.append(total_midi_notes)
       
    yolo_pitch = np.array(yolo_pitch)
    yolo_res = np.delete(yolo_res, del_list, axis=0)
    yolo_res[:, :2] *= ttime
    
    yolo_res[:, 2] = yolo_pitch
    pitch_list = [pitch + cfg['common_config']['midi_shift'] for pitch in pitch_list]

    assert len(yolo_res) == len(
        pitch_list), 'note length is not equal to pitch list length!'

    return yolo_res