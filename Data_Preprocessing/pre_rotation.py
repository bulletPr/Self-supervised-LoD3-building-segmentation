#
#
#      0=============================================================0
#      |    Project Name: Self-Supervised LoD3 Building Segmentation              |
#      0=============================================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements rotate point cloud
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2021/3/29 10:36
#
#


# ----------------------------------------
# Import packages and constant
# ----------------------------------------

def get_rotate_preprocess(create_labels=True):
  """Returns a function that does 90deg rotations and sets according labels."""

  def _four_rots(img):
    # We use our own instead of tf.image.rot90 because that one broke
    # internally shortly before deadline...
    return tf.stack([
        img,
        tf.transpose(tf.reverse_v2(img, [1]), [1, 0, 2]),
        tf.reverse_v2(img, [0, 1]),
        tf.reverse_v2(tf.transpose(img, [1, 0, 2]), [1]),
    ])

  def _rotate_pp(data):
    # Create labels in the same structure as images!
    if create_labels:
      data["label"] = utils.tf_apply_to_image_or_images(
          lambda _: tf.constant([0, 1, 2, 3]), data["image"], dtype=tf.int32)
    data["image"] = utils.tf_apply_to_image_or_images(_four_rots, data["image"])
    return data

  return _rotate_pp