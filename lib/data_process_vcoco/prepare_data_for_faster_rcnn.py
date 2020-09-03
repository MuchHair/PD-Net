import os

import lib.utils.io as io


def prepare_vcoco(coco_image_dir):
    exp_dir = "data/vcoco/"
    io.mkdir_if_not_exists(exp_dir)

    print('Loading anno_list.json ...')
    anno_list = io.load_json_object("data/vcoco/annotations/anno_list.json")

    print('Creating input json for faster rcnn ...')
    images_in_out = [None] * len(anno_list)
    for i, anno in enumerate(anno_list):
        global_id = anno['global_id']

        print(global_id)
        image_in_out = dict()
        image_in_out['in_path'] = os.path.join(
            coco_image_dir,
            anno['image_path_postfix'])
        assert os.path.exists(os.path.join(
            coco_image_dir, anno['image_path_postfix']))

        image_in_out['out_dir'] = os.path.join(
            exp_dir,
            'faster_rcnn_boxes')
        image_in_out['prefix'] = f'{global_id}_'
        images_in_out[i] = image_in_out

    images_in_out_json = os.path.join(
        exp_dir,
        'faster_rcnn_im_in_out.json')
    io.dump_json_object(images_in_out, images_in_out_json)


# python -m lib.data_process_vcoco.prepare_data_for_faster_rcnn
if __name__ == "__main__":
    prepare_vcoco(coco_image_dir="/home/xian/Documents/data/coco/images/")
