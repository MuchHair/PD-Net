from lib.utils import io as io
import h5py
from tqdm._tqdm import tqdm
import os


def generate_pkl_nis(pred_dets_hdf5, best_binary_file, out_dir, file_name):
    pred_dets = h5py.File(pred_dets_hdf5, 'r')
    binary_file = h5py.File(best_binary_file, 'r')
    print(pred_dets_hdf5)
    print(best_binary_file)

    assert len(pred_dets.keys()) == 4539

    print(len(binary_file))

    hoi_list = io.load_json_object("data/vcoco/annotations/hoi_list_234.json")
    hoi_dict = {int(hoi["id"]) - 1: hoi for hoi in hoi_list}

    result_list = []
    for global_id in tqdm(pred_dets.keys()):
        image_id = int(global_id.split("_")[1])

        start_end_ids = pred_dets[global_id]['start_end_ids']
        assert len(start_end_ids) == 234

        for hoi_id in range(234):
            start_id, end_id = pred_dets[global_id]['start_end_ids'][int(hoi_id)]
            if start_id == end_id:
                continue

            for j in range(start_id, end_id):
                hoi_dets = pred_dets[global_id]['human_obj_boxes_scores'][j]
                inter_score = binary_file[global_id]["binary_score_data"][j]

                final_score = hoi_dets[8] * inter_score * hoi_dets[9]
                person_boxes = hoi_dets[:4].tolist()

                per_image_dict = {}
                per_image_dict["image_id"] = image_id
                per_image_dict["person_box"] = person_boxes

                aciton = hoi_dict[hoi_id]["verb"]
                role = hoi_dict[hoi_id]["role"]

                per_image_dict[aciton + "_" + role] = [hoi_dets[4], hoi_dets[5],
                                                       hoi_dets[6], hoi_dets[7],
                                                       final_score]

                result_list.append(per_image_dict)

    io.dump_pickle_object(result_list, os.path.join(out_dir, file_name + ".pkl"))


# python -m eval.generate_pkl_nis
if __name__ == "__main__":
    generate_pkl_nis(
        "output/vcoco/PD/pred_hoi_dets_test_8000.hdf5",
        "output/vcoco/nis/pred_hoi_dets_test_32000.hdf5",
        "output/vcoco/PD/",
        "8000_nis_32000")
