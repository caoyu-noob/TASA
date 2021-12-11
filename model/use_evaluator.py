import tensorflow as tf
import tensorflow_hub as hub

class USEEvaluator(object):
    def __init__(self, use_model_path):
        super(USEEvaluator, self).__init__()
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        self.embed = hub.load(use_model_path)

    def cal_use_feature(self, sents):
        with tf.device("/gpu:0"):
            embed = self.embed(sents)["outputs"]
            embed = tf.stop_gradient(tf.nn.l2_normalize(embed, axis=1))
            return embed

    def semantic_sim(self, embed1, embed2):
        with tf.device("/gpu:0"):
            cosine_similarity = tf.reduce_sum(tf.multiply(embed1, embed2), axis=1)
            cosine_similarity = tf.clip_by_value(cosine_similarity, -1.0, 1.0)
            return (1.0 - tf.acos(cosine_similarity)).cpu().numpy()

    def get_semantic_sim_by_ref(self, ref, sents):
        embed_ref = self.cal_use_feature([ref])
        embeds = self.cal_use_feature(sents)
        return self.semantic_sim(embed_ref, embeds)

if __name__ == "__main__":
    use_model = USEEvaluator("../../model_hubs/USE_model")
    x = use_model.get_semantic_sim_by_ref("i like this", ["i like this", "i do not like this"])
    print("11")