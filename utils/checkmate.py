import os
import glob
import json
import numpy as np
import tensorflow as tf


class BestCheckpointSaver(object):
    def __init__(self, save_dir, num_to_keep=1, maximize=True, saver=None):
        self._num_to_keep = num_to_keep
        self._save_dir = save_dir
        self._save_path = os.path.join(save_dir, 'model')
        self._maximize = maximize
        self._saver = saver if saver else tf.train.Saver(
            max_to_keep=None,
            save_relative_paths=True
        )

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.best_checkpoints_file = os.path.join(save_dir, 'best_checkpoints')

    def handle(self, value, sess, global_step):
        current_ckpt = 'model-{}'.format(global_step)
        value = float(value)
        if not os.path.exists(self.best_checkpoints_file):
            self._save_best_checkpoints_file({current_ckpt: value})
            self._saver.save(sess, self._save_path, global_step)
            return

        best_checkpoints = self._load_best_checkpoints_file()

        if len(best_checkpoints) < self._num_to_keep:
            best_checkpoints[current_ckpt] = value
            self._save_best_checkpoints_file(best_checkpoints)
            self._saver.save(sess, self._save_path, global_step)
            return

        if self._maximize:
            should_save = not all(current_best >= value
                                  for current_best in best_checkpoints.values())
        else:
            should_save = not all(current_best <= value
                                  for current_best in best_checkpoints.values())
        if should_save:
            best_checkpoint_list = self._sort(best_checkpoints)

            worst_checkpoint = os.path.join(self._save_dir,
                                            best_checkpoint_list.pop(-1)[0])
            self._remove_outdated_checkpoint_files(worst_checkpoint)
            self._update_internal_saver_state(best_checkpoint_list)

            best_checkpoints = dict(best_checkpoint_list)
            best_checkpoints[current_ckpt] = value
            self._save_best_checkpoints_file(best_checkpoints)

            self._saver.save(sess, self._save_path, global_step)

    def _save_best_checkpoints_file(self, updated_best_checkpoints):
        with open(self.best_checkpoints_file, 'w') as f:
            json.dump(updated_best_checkpoints, f, indent=3)

    def _remove_outdated_checkpoint_files(self, worst_checkpoint):
        os.remove(os.path.join(self._save_dir, 'checkpoint'))
        for ckpt_file in glob.glob(worst_checkpoint + '.*'):
            os.remove(ckpt_file)

    def _update_internal_saver_state(self, best_checkpoint_list):
        best_checkpoint_files = [
            (ckpt[0], np.inf)  # TODO: Try to use actual file timestamp
            for ckpt in best_checkpoint_list
        ]
        self._saver.set_last_checkpoints_with_time(best_checkpoint_files)

    def _load_best_checkpoints_file(self):
        with open(self.best_checkpoints_file, 'r') as f:
            best_checkpoints = json.load(f)
        return best_checkpoints

    def _sort(self, best_checkpoints):
        best_checkpoints = [
            (ckpt, best_checkpoints[ckpt])
            for ckpt in sorted(best_checkpoints,
                               key=best_checkpoints.get,
                               reverse=self._maximize)
        ]
        return best_checkpoints


def get_best_checkpoint(best_checkpoint_dir, select_maximum_value=True):
    best_checkpoints_file = os.path.join(best_checkpoint_dir, 'best_checkpoints')
    assert os.path.exists(best_checkpoints_file)
    with open(best_checkpoints_file, 'r') as f:
        best_checkpoints = json.load(f)
    best_checkpoints = [
        ckpt for ckpt in sorted(best_checkpoints,
                                key=best_checkpoints.get,
                                reverse=select_maximum_value)
    ]
    return os.path.join(os.path.abspath(best_checkpoint_dir),  best_checkpoints[0])
