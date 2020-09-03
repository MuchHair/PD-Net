import vsrl_eval as eva
import argparse


#  python eval/eval_example.py --file
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")

    vcocoeval = eva.VCOCOeval('eval/data/vcoco_test.json', 'eval/data/instances_vcoco_all_2014.json',
                              'eval/data/vcoco_test.ids')

    this_output = parser.parse_args().file
    vcocoeval._do_eval(this_output, ovr_thresh=0.5, mode=1)
