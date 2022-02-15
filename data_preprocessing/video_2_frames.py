import utils
from data_preprocessing.colelaps_preprocessor import ColelapsPreprocessor


config = utils.get_config()
cholec80_path = config["dataset_path"]
frames_dest_path = config["frames_path"]
frames_index_path = "../results/videos_frames_index.csv"
preprocessor = ColelapsPreprocessor(cholec80_path=cholec80_path,
                                    dest_path=frames_dest_path,
                                    frames_index_path=frames_index_path)

videos_to_process = ['video01', 'video02', 'video03', 'video04', 'video05', 'video06',
                     'video07', 'video08', 'video09', 'video10', 'video11', 'video12',
                     'video13', 'video14', 'video15', 'video16', 'video17', 'video18',
                     'video19', 'video20', 'video21', 'video22', 'video23', 'video24',
                     'video25', 'video26', 'video27', 'video28', 'video29', 'video30',
                     'video31', 'video32', 'video33', 'video34', 'video35', 'video36',
                     'video37', 'video38', 'video39', 'video40', 'video41', 'video42',
                     'video43', 'video44', 'video45', 'video46', 'video47', 'video48',
                     'video49', 'video50', 'video51', 'video52', 'video53', 'video54',
                     'video55', 'video56', 'video57', 'video58', 'video59', 'video60',
                     'video61', 'video62', 'video63', 'video64', 'video65', 'video66',
                     'video67', 'video68', 'video69', 'video70', 'video71', 'video72',
                     'video73', 'video74', 'video75', 'video76', 'video77', 'video78',
                     'video79', 'video80']

preprocessor.process_videos(video_list=videos_to_process)
print("done")
