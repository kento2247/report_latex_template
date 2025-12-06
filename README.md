# report_latex_template

日本語レポート用LaTeXテンプレート

## ファイル構成

```
.
├── main.tex          # メインファイル（タイトル・著者情報を編集）
├── section1.tex      # セクション1
├── section2.tex      # セクション2
├── section3.tex      # セクション3
├── abstract.tex      # 概要（必要に応じて使用）
├── mymacros.sty      # カスタムマクロ定義
├── jsaiac.sty        # スタイルファイル
├── jsai.bst          # 参考文献スタイル
├── reference.bib     # 参考文献
├── latexmkrc         # latexmk設定
├── fig/              # 図を格納するディレクトリ
└── tab/              # 表を格納するディレクトリ
```

## 必要環境

- pLaTeX（platex）
- pBibTeX（pbibtex）
- dvipdfmx
- latexmk（推奨）

macOSの場合、MacTeXをインストールすれば全て揃います。

## コンパイル方法

### latexmkを使用（推奨）

```bash
latexmk main.tex
```

PDFを生成後、中間ファイルを削除する場合:

```bash
latexmk main.tex
latexmk -c
```

### 手動でコンパイル

```bash
platex main.tex
pbibtex main
platex main.tex
platex main.tex
dvipdfmx main.dvi
```

## 使い方

1. **タイトル・著者情報の編集**: `main.tex` の `\jtitle{}` と `\jname{}` を編集
2. **本文の編集**: `section1.tex`, `section2.tex`, `section3.tex` を編集
3. **セクションの追加**: 新しいファイルを作成し、`main.tex` に `\input{sectionX}` を追加
4. **図の挿入**: `fig/` ディレクトリに画像を配置し、`\includegraphics` で読み込み
5. **表の挿入**: `tab/` ディレクトリにテーブル定義を配置

## 便利なマクロ（mymacros.sty）

```latex
\figref{fig:label}    % → 図1 のように出力
\tabref{tab:label}    % → 表1 のように出力
\secref{sec:label}    % → 1節 のように出力
\eqref{eq:label}      % → 式(1) のように出力
```

## 参考文献の追加

参考文献を使用する場合は、以下の手順で設定します:

1. `reference.bib` に BibTeX エントリを追加
2. `main.tex` の `\end{document}` の前に以下を追加:

```latex
\bibliographystyle{jsai}
\bibliography{reference}
```

3. 本文中で `\cite{key}` で引用
