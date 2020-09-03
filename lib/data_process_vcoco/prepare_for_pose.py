import h5py
import tqdm
from lib.utils import io as io


def main(file_dir, subset):
    file = h5py.File(f"{file_dir}/hoi_candidates_{subset}.hdf5", "r")

    rpnidmap = {}
    for globalid in tqdm(file.keys()):
        assert globalid not in rpnidmap
        rpnidmap[globalid] = {}
        datas = file[globalid]["boxes_scores_rpn_ids_hoi_idx"]
        for data in datas:
            score = data[8]
            rpn = data[10]
            if rpn not in rpnidmap[globalid]:
                rpnidmap[globalid][rpn] = {
                    "score": score,
                    "box": [data[0], data[1], data[2], data[3]]
                }
            else:
                assert rpnidmap[globalid][rpn]["score"] == score
        print(len(rpnidmap[globalid]))

    io.dump_json_object(rpnidmap, f"{file_dir}/{subset}_human_boxes.json")


# python -m lib.data_process.prepare_for_pose
if __name__ == "__main__":
    for subset in ["train", "val", "test"]:
        main("data/vcoco",  f"{subset}")