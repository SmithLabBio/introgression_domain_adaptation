import numpy as np
import tensorflow as tf
from scipy.optimize import nnls
from adapt.feature_based import CDAN

EPS = np.finfo(np.float32).eps


class IWCDAN(CDAN):
    """
    CDAN with importance-weighted source loss for generalized label shift (Combes et al. 2020).
    """
    def __init__(self,
                 encoder=None,
                 task=None,
                 discriminator=None,
                 Xt=None,
                 yt=None,
                 lambda_=1.,
                 entropy=True,
                 max_features=4096,
                 verbose=1,
                 copy=True,
                 random_state=None,
                 n_classes=2,
                 **params):

        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)

        self.n_classes = n_classes
        self.class_weights_var = tf.Variable(
            tf.ones(n_classes), trainable=False, dtype=tf.float32, name="iw_class_weights")
        self.balance_weights_var = tf.Variable(
            tf.ones(n_classes), trainable=False, dtype=tf.float32, name="balance_weights")


    def set_balance_weights(self, y_source_int):
        """Compute BER weights once. Do not update during training. Must be called!."""
        counts = np.bincount(y_source_int, minlength=self.n_classes).astype(np.float32)
        p_source = counts / counts.sum()
        w = np.divide(1.0, p_source, out=np.ones_like(p_source), where=p_source > 0)
        w = w / w.mean()  # normalize so weights don't shift task loss's overall scale
        self.balance_weights_var.assign(w.astype(np.float32))


    def train_step(self, data):
        Xs, Xt, ys, yt = self._unpack_data(data)

        with tf.GradientTape() as task_tape, tf.GradientTape() as enc_tape, tf.GradientTape() as disc_tape:

            Xs_enc = self.encoder_(Xs, training=True)
            ys_pred = self.task_(Xs_enc, training=True)
            Xt_enc = self.encoder_(Xt, training=True)
            yt_pred = self.task_(Xt_enc, training=True)

            src_class_idx = tf.argmax(ys, axis=1)

            # importance weights
            iw_w = tf.gather(self.class_weights_var, src_class_idx)
            iw_w = iw_w / (tf.reduce_mean(iw_w) + EPS)

            # BER weights
            ber_w = tf.gather(self.balance_weights_var, src_class_idx)

            if self.is_overloaded_:
                mapping_task_src = tf.matmul(ys_pred, self._random_task)
                mapping_enc_src = tf.matmul(Xs_enc, self._random_enc)
                mapping_src = tf.multiply(mapping_enc_src, mapping_task_src)
                mapping_src /= (tf.math.sqrt(tf.cast(self.max_features, tf.float32)) + EPS)

                mapping_task_tgt = tf.matmul(yt_pred, self._random_task)
                mapping_enc_tgt = tf.matmul(Xt_enc, self._random_enc)
                mapping_tgt = tf.multiply(mapping_enc_tgt, mapping_task_tgt)
                mapping_tgt /= (tf.math.sqrt(tf.cast(self.max_features, tf.float32)) + EPS)
            else:
                mapping_src = tf.matmul(tf.expand_dims(Xs_enc, 2), tf.expand_dims(ys_pred, 1))
                mapping_tgt = tf.matmul(tf.expand_dims(Xt_enc, 2), tf.expand_dims(yt_pred, 1))
                dim = int(np.prod(mapping_src.get_shape()[1:]))
                mapping_src = tf.reshape(mapping_src, (-1, dim))
                mapping_tgt = tf.reshape(mapping_tgt, (-1, dim))

            ys_disc = self.discriminator_(mapping_src)
            yt_disc = self.discriminator_(mapping_tgt)

            if self.entropy:
                entropy_src = -tf.reduce_sum(ys_pred * tf.math.log(ys_pred + EPS), axis=1, keepdims=True)
                entropy_tgt = -tf.reduce_sum(yt_pred * tf.math.log(yt_pred + EPS), axis=1, keepdims=True)
                weight_src = 1. + tf.exp(-entropy_src)
                weight_tgt = 1. + tf.exp(-entropy_tgt)
                weight_src /= (tf.reduce_mean(weight_src) + EPS)
                weight_tgt /= (tf.reduce_mean(weight_tgt) + EPS)
                weight_src *= .5
                weight_tgt *= .5

                weight_src = weight_src * iw_w[:, tf.newaxis]

                disc_loss = (-weight_src * tf.math.log(ys_disc + EPS)
                             -weight_tgt * tf.math.log(1 - yt_disc + EPS))
            else:
                disc_loss = (-iw_w[:, tf.newaxis] * tf.math.log(ys_disc + EPS)
                             -tf.math.log(1 - yt_disc + EPS))

            ys_pred = tf.reshape(ys_pred, tf.shape(ys))

            task_loss_per_example = self.task_loss_(ys, ys_pred)
            task_loss = tf.reduce_mean(task_loss_per_example * ber_w)

            disc_loss = tf.reduce_mean(disc_loss)
            enc_loss = task_loss - self.lambda_ * disc_loss

            task_loss += sum(self.task_.losses)
            disc_loss += sum(self.discriminator_.losses)
            enc_loss += sum(self.encoder_.losses)

        trainable_vars_task = self.task_.trainable_variables
        trainable_vars_enc = self.encoder_.trainable_variables
        trainable_vars_disc = self.discriminator_.trainable_variables

        gradients_task = task_tape.gradient(task_loss, trainable_vars_task)
        gradients_enc = enc_tape.gradient(enc_loss, trainable_vars_enc)
        gradients_disc = disc_tape.gradient(disc_loss, trainable_vars_disc)

        self.optimizer.apply_gradients(zip(gradients_task, trainable_vars_task))
        self.optimizer_disc.apply_gradients(zip(gradients_disc, trainable_vars_disc))
        if len(gradients_enc) > 0:
            self.optimizer_enc.apply_gradients(zip(gradients_enc, trainable_vars_enc))

        logs = self._update_logs(ys, ys_pred)
        disc_metrics = self._get_disc_metrics(ys_disc, yt_disc)
        logs.update({"disc_loss": disc_loss, "mean_iw": tf.reduce_mean(iw_w)})
        logs.update(disc_metrics)
        return logs

class IWWeightUpdater(tf.keras.callbacks.Callback):
    """
    Estimate importance weights.
    """
    def __init__(self, iwcdan_model, Xs_heldout, ys_heldout, Xt,
                 update_every=5, min_weight=0.2, max_weight=5.0, momentum=0.5, verbose=True):
        super().__init__()
        self.iwcdan_model = iwcdan_model
        self.Xs_heldout = Xs_heldout
        self.ys_heldout = ys_heldout
        self.Xt = Xt
        self.update_every = update_every
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.momentum = momentum
        self.verbose = verbose
        self.n_classes = int(ys_heldout.max()) + 1

    def _predict_probs(self, X):
        enc = self.iwcdan_model.encoder_.predict(X, verbose=0)
        return self.iwcdan_model.task_.predict(enc, verbose=0)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.update_every != 0:
            return

        probs_s = self._predict_probs(self.Xs_heldout)
        probs_t = self._predict_probs(self.Xt)

        # soft confusion matrix: C[i, j] = mean predicted prob of class i, over true class j
        C = np.zeros((self.n_classes, self.n_classes))
        for j in range(self.n_classes):
            mask = self.ys_heldout == j
            if mask.sum() > 0:
                C[:, j] = probs_s[mask].mean(axis=0)

        mu_hat = probs_t.mean(axis=0)

        p_target, _ = nnls(C, mu_hat)
        if p_target.sum() <= 0:
            if self.verbose:
                print(f"[IW update epoch {epoch+1}] degenerate estimate, skipping update")
            return
        p_target = p_target / p_target.sum()

        p_source = np.bincount(self.ys_heldout, minlength=self.n_classes) / len(self.ys_heldout)

        new_weights = np.divide(p_target, p_source, out=np.ones_like(p_target), where=p_source > 0)
        new_weights = np.clip(new_weights, self.min_weight, self.max_weight)

        old_weights = self.iwcdan_model.class_weights_var.numpy()
        smoothed_weights = self.momentum * old_weights + (1 - self.momentum) * new_weights
        self.iwcdan_model.class_weights_var.assign(smoothed_weights.astype(np.float32))

        if self.verbose:
            print(f"[IW update epoch {epoch+1}] p_target~{p_target.round(3)}  "
                  f"raw~{new_weights.round(3)}  smoothed~{smoothed_weights.round(3)}")

