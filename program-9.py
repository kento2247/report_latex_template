# -*- coding: utf-8 -*-

# ヘッブの学習（主成分分析）- Sangerの学習則
# MNISTデータセット用

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


class HebbianPCA:
    def __init__(self, input_size=28 * 28, output_size=10, alpha=0.01, loop=1000):
        """
        ヘッブの学習（Sangerの学習則）によるPCA

        Parameters:
        -----------
        input_size : int
            入力の次元数（画像のピクセル数）
        output_size : int
            出力の個数（固有ベクトルの個数 m）
        alpha : float
            学習係数
        loop : int
            学習回数
        """
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        self.loop = loop

        # 重みの初期化
        self.weight = np.random.uniform(-0.5, 0.5, (output_size, input_size))

        # 重みの変更値
        self.d_weight = np.zeros((output_size, input_size))

        # 学習データの平均（復元時に使用）
        self.mean = None

    def load_mnist_data(self, data_dir, train_num=100):
        """
        MNISTデータの読み込み

        Parameters:
        -----------
        data_dir : str
            データディレクトリのパス
        train_num : int
            各クラスの学習枚数

        Returns:
        --------
        train_vec : np.ndarray
            学習データ (全データ数, input_size)
        """
        class_num = 10  # 0-9の10クラス
        train_vec = []

        for i in range(class_num):
            class_dir = os.path.join(data_dir, str(i))
            files = sorted(os.listdir(class_dir))[:train_num]

            for file in files:
                file_path = os.path.join(class_dir, file)
                img = Image.open(file_path).convert("L")
                img_array = np.asarray(img).astype(np.float64).flatten()
                train_vec.append(img_array)

        train_vec = np.array(train_vec)

        # 平均を保存（復元時に使用）
        self.mean = np.mean(train_vec, axis=0)

        # 平均を引いて中心化
        train_vec = train_vec - self.mean

        # ノルムを1に正規化
        norms = np.linalg.norm(train_vec, axis=1, keepdims=True)
        norms[norms == 0] = 1  # ゼロ除算を防ぐ
        train_vec = train_vec / norms

        return train_vec

    def train(self, train_vec):
        """
        Sangerの学習則によるヘッブ学習（ベクトル化版）

        Parameters:
        -----------
        train_vec : np.ndarray
            学習データ (データ数, input_size)
        """
        train_num = train_vec.shape[0]

        # 下三角行列（Sangerの学習則で使用）
        lower_triangular = np.tril(np.ones((self.output_size, self.output_size)))

        for _ in tqdm(range(self.loop), desc="Training"):
            for t in range(train_num):
                # 入力ベクトル (input_size,)
                e = train_vec[t]

                # 出力値の計算 V = W @ e, shape: (output_size,)
                V = np.dot(self.weight, e)

                # Sangerの学習則（ベクトル化）
                # sum_o[i] = Σ_{k=0}^{i} V[k] * W[k]
                # 下三角行列を使って累積和を計算
                # VW[k] = V[k] * W[k], shape: (output_size, input_size)
                VW = V[:, np.newaxis] * self.weight

                # sum_o[i] = Σ_{k=0}^{i} VW[k], shape: (output_size, input_size)
                sum_o = np.dot(lower_triangular, VW)

                # d_weight[i] = alpha * V[i] * (e - sum_o[i])
                self.d_weight = self.alpha * V[:, np.newaxis] * (e - sum_o)

                # 重みの更新
                self.weight += self.d_weight

        print("Training completed!")

    def transform(self, data):
        """
        データを主成分空間に射影

        Parameters:
        -----------
        data : np.ndarray
            入力データ (データ数, input_size) または (input_size,)

        Returns:
        --------
        projected : np.ndarray
            主成分空間に射影されたデータ
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # 中心化
        centered = data - self.mean

        # 射影
        projected = np.dot(centered, self.weight.T)

        return projected

    def reconstruct(self, data):
        """
        主成分から元の画像を復元

        Parameters:
        -----------
        data : np.ndarray
            入力データ (データ数, input_size) または (input_size,)

        Returns:
        --------
        reconstructed : np.ndarray
            復元された画像データ
        """
        # 主成分空間に射影
        projected = self.transform(data)

        # 復元: x_reconstructed = projected @ W + mean
        reconstructed = np.dot(projected, self.weight) + self.mean

        return reconstructed

    def save_weight_images(self, output_dir, size=28):
        """
        重みベクトル（固有ベクトル）を画像として保存

        Parameters:
        -----------
        output_dir : str
            出力ディレクトリ
        size : int
            画像のサイズ（size x size）
        """
        os.makedirs(output_dir, exist_ok=True)

        for j in range(self.output_size):
            a = np.reshape(self.weight[j], (size, size))
            plt.figure()
            plt.imshow(a, cmap="gray", interpolation="nearest")
            plt.colorbar()
            plt.title(f"Eigenvector {j+1}")
            file = os.path.join(output_dir, f"weight-{j}.png")
            plt.savefig(file)
            plt.close()

        print(f"Weight images saved to {output_dir}")

    def save_weights(self, filename):
        """
        重みと平均を保存

        Parameters:
        -----------
        filename : str
            保存ファイル名
        """
        np.savez(filename, weight=self.weight, mean=self.mean)
        print(f"Weights saved to {filename}")

    def load_weights(self, filename):
        """
        重みと平均を読み込み

        Parameters:
        -----------
        filename : str
            読み込みファイル名
        """
        data = np.load(filename)
        self.weight = data["weight"]
        self.mean = data["mean"]
        print(f"Weights loaded from {filename}")

    def check_orthogonality(self):
        """
        重みベクトル（固有ベクトル）の直交性を確認
        """
        print("\n[直交性の確認]")
        for i in range(self.output_size):
            for j in range(i, self.output_size):
                dot_product = np.dot(self.weight[i], self.weight[j])
                print(f"w{i} · w{j} = {dot_product:.6f}")


def visualize_reconstruction(pca, data_dir, output_dir, num_samples=5, size=28):
    """
    元画像と復元画像を比較表示

    Parameters:
    -----------
    pca : HebbianPCA
        学習済みPCAモデル
    data_dir : str
        データディレクトリ
    output_dir : str
        出力ディレクトリ
    num_samples : int
        表示するサンプル数
    size : int
        画像サイズ
    """
    os.makedirs(output_dir, exist_ok=True)

    # テストデータを読み込み（各クラスから1枚ずつ）
    test_images = []
    labels = []

    for i in range(10):
        class_dir = os.path.join(data_dir, str(i))
        files = sorted(os.listdir(class_dir))
        if files:
            file_path = os.path.join(class_dir, files[0])
            img = Image.open(file_path).convert("L")
            img_array = np.asarray(img).astype(np.float64).flatten()
            test_images.append(img_array)
            labels.append(i)

    test_images = np.array(test_images)

    # 復元
    reconstructed = pca.reconstruct(test_images)

    # 可視化
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))

    for i in range(10):
        # 元画像
        axes[0, i].imshow(test_images[i].reshape(size, size), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"{labels[i]}")

        # 復元画像
        axes[1, i].imshow(reconstructed[i].reshape(size, size), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=10)

    plt.suptitle(f"Image Reconstruction using {pca.output_size} Principal Components")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reconstruction_comparison.png"), dpi=150)
    plt.close()

    print(
        f"Reconstruction comparison saved to {output_dir}/reconstruction_comparison.png"
    )

    # 復元誤差を計算
    mse = np.mean((test_images - reconstructed) ** 2)
    print(f"Mean Squared Error: {mse:.4f}")


if __name__ == "__main__":
    # パラメータ
    input_size = 28 * 28  # MNIST画像サイズ
    output_size = 10  # 固有ベクトルの個数 m
    alpha = 0.01  # 学習係数
    loop = 500  # 学習回数
    train_num = 100  # 各クラスの学習枚数

    # データディレクトリ
    data_dir = "data/mnist/train"
    output_dir = "program-8/fig"

    # PCAモデルの初期化
    pca = HebbianPCA(
        input_size=input_size, output_size=output_size, alpha=alpha, loop=loop
    )

    # 学習データの読み込み
    print("Loading MNIST training data...")
    train_vec = pca.load_mnist_data(data_dir, train_num=train_num)
    print(f"Loaded {train_vec.shape[0]} images")

    # 学習
    print("\nStarting Hebbian learning (Sanger's rule)...")
    pca.train(train_vec)

    # 重みベクトルの画像化
    pca.save_weight_images(output_dir)

    # 重みの保存
    pca.save_weights("program-8/weight-pca-mnist.npz")

    # 直交性の確認
    pca.check_orthogonality()

    # 復元の確認
    print("\n[復元の確認]")
    visualize_reconstruction(pca, data_dir, output_dir)
