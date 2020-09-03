import lib.utils.io as io


def convert(in_path, out_path):
    pose_dict = io.load_json_object(in_path)

    posedict = {}
    for item in pose_dict:
        imageid = item["image_id"]
        keypoints = item["keypoints"]
        rpnid = item["idx"]
        if imageid not in posedict:
            posedict[imageid] = {}
        assert len(keypoints) == 51
        assert int(rpnid) not in posedict[imageid]
        posedict[imageid][int(rpnid)] = {
            "keypoints": keypoints,
        }

    print(len(posedict))
    io.dump_json_object(posedict, out_path)


# python -m lib.data_process_hico.cpn_convert
if __name__ == "__main__":
    for subset in ["train", "val", "test"]:
        convert(f"/home/xian/Documents/code/AlphaPose/output/vcoco/vcoco/{subset}/alphapose-results.json",
                f"data/vcoco/alphapose-results_{subset}.json")

