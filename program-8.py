# -*- coding: utf-8 -*-
"""
3層型ニューラルネットワークを用いて，
❑ data/cifar-10/train/以下の画像（2,000枚）を学習
❑ data/cifar-10/test/以下の画像（2,000枚）を認識
しなさい．
"""

import argparse
import os

import numpy as np
from PIL import Image


class ThreeLayerNN:
    def __init__(
        self, input_size=32 * 32 * 3, hidden_size=128, output_size=10, alpha=0.01
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha

        # 中間層の重みと閾値
        self.w1 = np.random.uniform(-0.5, 0.5, (input_size, hidden_size)) * np.sqrt(
            2.0 / input_size
        )
        self.b1 = np.zeros(hidden_size)

        # 出力層の重みと閾値
        self.w2 = np.random.uniform(-0.5, 0.5, (hidden_size, output_size)) * np.sqrt(
            2.0 / hidden_size
        )
        self.b2 = np.zeros(output_size)

        # クラス名
        self.class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        # 中間層
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)

        # 出力層
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2

    def backward(self, x, y):
        batch_size = x.shape[0]

        # 出力層の誤差
        delta2 = self.a2 - y
        grad_w2 = np.dot(self.a1.T, delta2) / batch_size
        grad_b2 = np.sum(delta2, axis=0) / batch_size

        # 中間層の誤差
        delta1 = np.dot(delta2, self.w2.T) * self.relu_derivative(self.z1)
        grad_w1 = np.dot(x.T, delta1) / batch_size
        grad_b1 = np.sum(delta1, axis=0) / batch_size

        # 重みの更新
        self.w2 -= self.alpha * grad_w2
        self.b2 -= self.alpha * grad_b2
        self.w1 -= self.alpha * grad_w1
        self.b1 -= self.alpha * grad_b1

    def load_data(self, data_dir, train_num=200):
        """データの読み込み"""
        data = []
        labels = []

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            files = sorted(os.listdir(class_dir))[:train_num]

            for file in files:
                file_path = os.path.join(class_dir, file)
                img = Image.open(file_path).convert("RGB")
                img_array = np.asarray(img).astype(np.float64).flatten()
                img_array = img_array / 255.0  # 正規化
                data.append(img_array)
                labels.append(class_idx)

        return np.array(data), np.array(labels)

    def train(self, train_dir, epochs=100, batch_size=32, train_num=200):
        """学習"""
        print("Loading training data...")
        X_train, y_train = self.load_data(train_dir, train_num)

        # one-hotエンコーディング
        y_onehot = np.zeros((len(y_train), self.output_size))
        y_onehot[np.arange(len(y_train)), y_train] = 1

        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # シャッフル
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_onehot[indices]

            total_loss = 0
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # 順伝播
                output = self.forward(X_batch)

                # 逆伝播
                self.backward(X_batch, y_batch)

                # 損失計算
                loss = -np.sum(y_batch * np.log(output + 1e-8)) / len(X_batch)
                total_loss += loss

            if (epoch + 1) % 10 == 0:
                # 精度計算
                pred = self.forward(X_train)
                accuracy = np.mean(np.argmax(pred, axis=1) == y_train)
                print(
                    f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}"
                )

        print("Training completed!")

    def predict(self, test_dir, train_num=200):
        """予測"""
        print("Loading test data...")
        X_test, y_test = self.load_data(test_dir, train_num)

        # 予測
        output = self.forward(X_test)
        predictions = np.argmax(output, axis=1)

        # 混同行列
        confusion_matrix = np.zeros(
            (self.output_size, self.output_size), dtype=np.int32
        )
        for true_label, pred_label in zip(y_test, predictions):
            confusion_matrix[true_label][pred_label] += 1

        print("\n[混同行列]")
        print("予測→", end="")
        for name in self.class_names:
            print(f"{name[:4]:>5}", end="")
        print()

        for i, name in enumerate(self.class_names):
            print(f"{name[:10]:>10}", end="")
            for j in range(self.output_size):
                print(f"{confusion_matrix[i][j]:>5}", end="")
            print()

        print(f"\n正解数 -> {np.trace(confusion_matrix)} / {len(y_test)}")
        print(f"正解率 -> {np.trace(confusion_matrix) / len(y_test) * 100:.2f}%")

        return confusion_matrix

    def save(self, filename):
        """モデルの保存"""
        np.savez(filename, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
        print(f"Model saved to {filename}")

    def load(self, filename):
        """モデルの読み込み"""
        data = np.load(filename)
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
        print(f"Model loaded from {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3層型ニューラルネットワークによるCIFAR-10分類"
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["train", "predict", "both"],
        default="both",
        help="実行モード: train, predict, both (default: both)",
    )
    parser.add_argument(
        "--input-size", type=int, default=32 * 32 * 3, help="入力サイズ (default: 3072)"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=256, help="中間層のサイズ (default: 256)"
    )
    parser.add_argument(
        "--output-size", type=int, default=10, help="出力サイズ (default: 10)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01, help="学習率 (default: 0.01)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="エポック数 (default: 100)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="バッチサイズ (default: 32)"
    )
    parser.add_argument(
        "--train-num", type=int, default=200, help="各クラスの学習枚数 (default: 200)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/cifar-10",
        help="データディレクトリ (default: data/cifar-10)",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default="program-8/model.npz",
        help="モデルファイルのパス (default: program-8/model.npz)",
    )
    args = parser.parse_args()

    # データディレクトリ
    train_dir = os.path.join(args.base_dir, "train")
    test_dir = os.path.join(args.base_dir, "test")

    # ニューラルネットワークの初期化
    nn = ThreeLayerNN(args.input_size, args.hidden_size, args.output_size, args.alpha)

    if args.mode == "train":
        # 学習のみ
        nn.train(train_dir, args.epochs, args.batch_size, args.train_num)
        nn.save(args.model_file)
    elif args.mode == "predict":
        # 予測のみ
        nn.load(args.model_file)
        nn.predict(test_dir, args.train_num)
    else:
        # 学習→予測
        nn.train(train_dir, args.epochs, args.batch_size, args.train_num)
        nn.save(args.model_file)
        nn.predict(test_dir, args.train_num)
