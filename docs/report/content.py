"""Content of the final report (journal style), kept apart from the layout code.

All scores, counts and example sentences are read from results/ so that the report cannot drift
from the experiments. build_report.py turns this content into report.tex and report.pdf.

Inline markup used in the text strings:
    *italic*   **bold**   `code`        [@key] (Author Year)    [+key] Author (Year)
    [[tab:x]] / [[fig:x]]  reference to a table or figure       -- en dash
"""

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from compare import count_categories, normalise_category  # noqa: E402

RESULTS = ROOT / "results"


def score(name):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def d2(x):
    return f"{x:.2f}"


def sci(x):
    """3e-05 -> 3e-5"""
    return f"{x:.0e}".replace("e-0", "e-")


def eda_stats():
    """Parse results/eda.txt: split sizes, sentence lengths and references without a real word."""
    text = (RESULTS / "eda.txt").read_text(encoding="utf-8")
    stats = {}
    for m in re.finditer(r"^(train|val|test)\s+n=\s*(\d+)\s+src_len mean=([\d.]+) max=(\d+)\s+"
                         r"tgt_len mean=([\d.]+) max=(\d+)\s+word-overlap mean=([\d.]+)", text, re.M):
        stats[m.group(1)] = dict(n=int(m.group(2)), src_mean=m.group(3), src_max=m.group(4),
                                 tgt_mean=m.group(5), tgt_max=m.group(6), overlap=m.group(7))
    for m in re.finditer(r"^\s+(train|val|test)\s+(\d+) of (\d+) \(([\d.]+)%\)", text, re.M):
        stats[m.group(1)]["junk"] = int(m.group(2))
        stats[m.group(1)]["junk_pct"] = m.group(4)
    return stats


def lookup(errors_name, source):
    """Return (prediction, reference, category) of a real row of an errors csv."""
    with open(RESULTS / f"{errors_name}.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["source"] == source:
                return row["prediction"], row["reference"], normalise_category(row["error_category"] or "")
    raise KeyError(f"{source!r} not found in {errors_name}.csv")


REFERENCES = {
    "bahdanau2015": ("Bahdanau et al.", "2015",
        "Bahdanau, D., Cho, K. and Bengio, Y. (2015) Neural machine translation by jointly learning to "
        "align and translate. *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*."),
    "gala2023": ("Gala et al.", "2023",
        "Gala, J. *et al.* (2023) IndicTrans2: towards high-quality and accessible machine translation "
        "models for all 22 scheduled Indian languages. *Transactions on Machine Learning Research*."),
    "hochreiter1997": ("Hochreiter and Schmidhuber", "1997",
        "Hochreiter, S. and Schmidhuber, J. (1997) Long short-term memory. *Neural Computation* **9**(8): 1735--1780."),
    "kingma2015": ("Kingma and Ba", "2015",
        "Kingma, D. P. and Ba, J. (2015) Adam: a method for stochastic optimization. "
        "*Proceedings of the 3rd International Conference on Learning Representations (ICLR)*."),
    "kudo2018": ("Kudo and Richardson", "2018",
        "Kudo, T. and Richardson, J. (2018) SentencePiece: a simple and language independent subword "
        "tokenizer and detokenizer for neural text processing. *Proceedings of EMNLP 2018: System Demonstrations*, pp. 66--71."),
    "liu2020": ("Liu et al.", "2020",
        "Liu, Y., Gu, J., Goyal, N., Li, X., Edunov, S., Ghazvininejad, M., Lewis, M. and Zettlemoyer, L. (2020) "
        "Multilingual denoising pre-training for neural machine translation. "
        "*Transactions of the Association for Computational Linguistics* **8**: 726--742."),
    "papineni2002": ("Papineni et al.", "2002",
        "Papineni, K., Roukos, S., Ward, T. and Zhu, W.-J. (2002) BLEU: a method for automatic evaluation of "
        "machine translation. *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (ACL)*, pp. 311--318."),
    "paszke2019": ("Paszke et al.", "2019",
        "Paszke, A. *et al.* (2019) PyTorch: an imperative style, high-performance deep learning library. "
        "*Advances in Neural Information Processing Systems 32 (NeurIPS)*."),
    "popovic2015": ("Popović", "2015",
        "Popović, M. (2015) chrF: character n-gram F-score for automatic MT evaluation. "
        "*Proceedings of the Tenth Workshop on Statistical Machine Translation*, pp. 392--395."),
    "post2018": ("Post", "2018",
        "Post, M. (2018) A call for clarity in reporting BLEU scores. "
        "*Proceedings of the Third Conference on Machine Translation: Research Papers*, pp. 186--191."),
    "sennrich2016": ("Sennrich et al.", "2016",
        "Sennrich, R., Haddow, B. and Birch, A. (2016) Neural machine translation of rare words with subword units. "
        "*Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL)*, pp. 1715--1725."),
    "srivastava2020": ("Srivastava and Singh", "2020",
        "Srivastava, V. and Singh, M. (2020) PHINC: a parallel Hinglish social media code-mixed corpus for machine "
        "translation. *Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)*."),
    "szegedy2016": ("Szegedy et al.", "2016",
        "Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J. and Wojna, Z. (2016) Rethinking the Inception "
        "architecture for computer vision. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2818--2826."),
    "tang2020": ("Tang et al.", "2020",
        "Tang, Y., Tran, C., Li, X., Chen, P.-J., Goyal, N., Chaudhary, V., Gu, J. and Fan, A. (2020) Multilingual "
        "translation with extensible multilingual pretraining and finetuning. *arXiv preprint* arXiv:2008.00401."),
    "vaswani2017": ("Vaswani et al.", "2017",
        "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. and Polosukhin, I. (2017) "
        "Attention is all you need. *Advances in Neural Information Processing Systems 30 (NeurIPS)*."),
    "wolf2020": ("Wolf et al.", "2020",
        "Wolf, T. *et al.* (2020) Transformers: state-of-the-art natural language processing. "
        "*Proceedings of EMNLP 2020: System Demonstrations*."),
    "xue2021": ("Xue et al.", "2021",
        "Xue, L., Constant, N., Roberts, A., Kale, M., Al-Rfou, R., Siddhant, A., Barua, A. and Raffel, C. (2021) "
        "mT5: a massively multilingual pre-trained text-to-text transformer. "
        "*Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT)*."),
}


def build():
    """Return the document as a dict of metadata and a list of blocks."""
    lstm, tr = score("lstm_attn"), score("transformer")
    tr_small = score("transformer_small")
    mb, mbz = score("finetune_mbart"), score("finetune_mbart_zeroshot")
    mt, mtz = score("finetune_mt5"), score("finetune_mt5_zeroshot")
    st = eda_stats()
    b = lambda s: s["bleu"]  # noqa: E731
    c = lambda s: s["chrf"]  # noqa: E731
    n_test = lstm["num_sentences"]
    assert n_test == st["test"]["n"] == tr["num_sentences"] == mb["num_sentences"] == 1345

    cats = {
        "LSTM": count_categories(RESULTS / "lstm_attn_errors.csv"),
        "Transformer": count_categories(RESULTS / "transformer_errors.csv"),
        "mT5-small": count_categories(RESULTS / "finetune_mt5_errors.csv"),
        "mBART-50": count_categories(RESULTS / "finetune_mbart_errors.csv"),
    }
    order = {}
    for cnt in cats.values():
        for k, v in cnt.items():
            order[k] = order.get(k, 0) + v
    cat_names = sorted(order, key=lambda k: -order[k])
    cat_rows = [[k] + [str(cats[m].get(k, "")) for m in cats] for k in cat_names]
    cat_rows.append(["**Tagged rows**"] + [f"**{sum(cats[m].values())}**" for m in cats])

    ex = []
    for model, errors, src in [
        ("Transformer", "transformer_errors", "mobarak ied mobarak ??"),
        ("mBART-50", "finetune_mbart_errors", "mobarak ied mobarak ??"),
        ("mT5-small", "finetune_mt5_errors", "kya yr maine b mis kr diya #uhh"),
        ("LSTM", "lstm_attn_errors", "hamari bus main bhi ek dengu aata tha ,"),
        ("mBART-50", "finetune_mbart_errors", "haha kya haal chaal hai?"),
    ]:
        pred, ref, cat = lookup(errors, src)
        ex.append([f"`{src}`", f"`{ref}`", model, f"`{pred}`", cat])

    blocks = []
    P = lambda t: blocks.append(("p", t))  # noqa: E731
    H1 = lambda t: blocks.append(("h1", t))  # noqa: E731
    H2 = lambda t: blocks.append(("h2", t))  # noqa: E731

    # ---------------------------------------------------------------- 1 Introduction
    H1("Introduction")
    P("Social media users in India routinely write Hindi in the Latin alphabet and mix it with English "
      "inside a single sentence, for example *kya baat hai bhaiya* (“wow, brother”). This "
      "style, called Hinglish, is a form of code-mixing. It is hard for machine translation for three "
      "reasons: parallel data is scarce, Hindi words in Roman script have no fixed spelling (*kya*, *kyaa*, "
      "*kyaaa*), and many English words appear unchanged between the two languages, so a model has to "
      "decide for every word whether to translate it or to copy it.")
    P("Large pretrained multilingual models are the usual answer to scarce data, but they were not "
      "trained on Roman-script Hindi as a language of its own, and it is not clear how much they help "
      "compared with a smaller model trained from scratch on the little data that exists. This is the "
      "question of this project: **how much does pretraining help on a low-resource, code-mixed "
      "translation task compared with training from scratch?**")
    P("We study it on the PHINC corpus [@srivastava2020] with four systems that use one fixed data split "
      "and one scoring function: a recurrent LSTM with Bahdanau attention [@bahdanau2015] and a "
      "Transformer [@vaswani2017], both trained from scratch, and the pretrained mBART-50 [@tang2020] and "
      "mT5-small [@xue2021], each evaluated zero-shot and after fine-tuning. Our contributions are:")
    blocks.append(("ul", [
        "a cleaned, fixed train/validation/test split of PHINC, with the data problems we found in the raw "
        "file documented and fixed;",
        "a shared evaluation protocol (BLEU and chrF through one function) so that all systems are "
        "comparable;",
        f"results for six system configurations on {n_test:,} test sentences, with a hand-tagged error "
        "analysis of the 50 worst translations of each trained system;",
        "an open, tested code base with continuous integration, from which every number in this report "
        "can be regenerated."]))
    P("The rest of the paper is organised as follows. Section 2 gives background, Section 3 describes the "
      "data, Section 4 the systems and the evaluation, Section 5 the results, Section 6 the error "
      "analysis, and Sections 7 and 8 discuss limitations and conclude.")

    # ---------------------------------------------------------------- 2 Background
    H1("Background and related work")
    P("**Code-mixed translation.** Code-mixed text is noisy and low-resource, and parallel corpora for it "
      "are small. PHINC [@srivastava2020] is a parallel corpus of Hinglish social media sentences with "
      "English translations written by human annotators; it is the data used here.")
    P("**Neural translation from scratch.** Sequence-to-sequence models with attention "
      "[@bahdanau2015], built on LSTM networks [@hochreiter1997], learn to align target words with "
      "source words. The Transformer [@vaswani2017] replaces recurrence by self-attention and is the "
      "standard architecture today. Subword vocabularies such as byte-pair encoding "
      "[@sennrich2016], trained with SentencePiece [@kudo2018], let a model represent rare and "
      "misspelled words as sequences of known pieces.")
    P("**Pretrained multilingual models.** mBART [@liu2020] is pretrained by denoising text in many "
      "languages and, as mBART-50 [@tang2020], fine-tuned for translation between 50 languages. "
      "mT5 [@xue2021] is a multilingual text-to-text model pretrained on web text. IndicTrans2 "
      "[@gala2023] is a translation model for Indian languages that is a natural candidate for Hindi; we "
      "prepared code for it but did not run it (Section 7).")
    P("**Evaluation.** We report BLEU [@papineni2002] and chrF [@popovic2015], computed with "
      "sacreBLEU [@post2018]. BLEU compares word n-grams with the reference and is known to be harsh on "
      "short and noisy sentences; chrF compares character n-grams and is more forgiving of spelling "
      "variation, which suits Hinglish.")

    # ---------------------------------------------------------------- 3 Data
    H1("Data")
    H2("The PHINC corpus")
    P("PHINC contains 13,738 rows, each a Roman-script Hinglish sentence taken from social media and its "
      "English translation. We lowercase all text and replace URLs by the token `<url>`; mentions, "
      "hashtags and emoticons are kept because the English reference keeps them too.")
    H2("Problems in the raw file")
    P("Reviews of our own pipeline found problems in the raw data that would have distorted every "
      "result, and we fixed them before any final training:")
    blocks.append(("ul", [
        "**Junk translations.** 234 rows have an English side that is only an emoticon such as `)`, `$` "
        "or `-)`. A model trained on them learned to output `)`. Rows with no letter on either side are "
        "dropped.",
        "**Cut tweets.** 51 pairs come from tweets that were cut at the HTML code `&amp;`, so source and "
        "reference are two halves of one English sentence (for example `...from h&` and `m tomorrow`). "
        "Pairs where a side ends with a bare `&` are dropped.",
        "**HTML codes.** `&amp;` and `&lt;` are decoded.",
        "**Duplicates.** Exact duplicate (Hinglish, English) pairs are removed, so no identical pair is in "
        "two splits."]))
    H2("Fixed split")
    P("After cleaning, the data is shuffled with seed 42 and cut into 80% training, 10% validation and 10% "
      "test data ([[tab:data]]). All systems are trained on the training split, tuned on the validation "
      "split and evaluated once on the test split. Vocabularies and tokenizers are built from the training "
      "split only. About 28% of the Hinglish words also appear unchanged in the English sentence, which "
      "shows how much English is mixed in. The same Hinglish sentence can still occur with a different "
      "translation in two splits (one validation and one test sentence also occur in training).")
    P(f"The references are noisy. {st['test']['junk']} of the {st['test']['n']:,} test references "
      f"({st['test']['junk_pct']}%) contain no real word at all, only emoticons, mentions, hashtags or "
      "`<url>` (for example `) <url>`). We left them in so that all systems are scored on the same fixed "
      "split.")
    blocks.append(("table", dict(
        label="data", caption="Statistics of the fixed split (words per sentence, after cleaning)",
        cols=["Split", "Pairs", "Hinglish mean", "Hinglish max", "English mean", "English max"],
        align="lrrrrr",
        rows=[[nm, f"{st[k]['n']:,}", st[k]["src_mean"], st[k]["src_max"], st[k]["tgt_mean"], st[k]["tgt_max"]]
              for nm, k in (("Train", "train"), ("Validation", "val"), ("Test", "test"))])))

    # ---------------------------------------------------------------- 4 Systems
    H1("Systems and evaluation")
    H2("LSTM with Bahdanau attention")
    P("The first system is a recurrent encoder-decoder. A bidirectional LSTM reads the Hinglish sentence; "
      "an LSTM decoder produces the English sentence one word at a time and, at every step, attends over "
      "all source positions with additive (Bahdanau) attention. It uses word-level vocabularies of the "
      "words seen at least twice in the training data, embeddings of size 256, a decoder of size 512, "
      "dropout 0.3, the Adam optimiser [@kingma2015] with learning rate 0.001, label smoothing "
      "[@szegedy2016] of 0.1 and early stopping on validation loss. Training stopped after 15 epochs. "
      "About 12% of the English training words are rare and become the unknown token `<unk>`; the model "
      "therefore learned to output `<unk>`, and blocking it during decoding raised BLEU from 6.50 to "
      f"{d2(b(lstm))}. We decode with beam search of size 5.")
    blocks.append(("fig", dict(
        label="pipeline", file="figures/pipeline.png", width=0.98,
        caption="Overview of the experimental pipeline. All systems read the same split and are scored by "
                "the same evaluation function.")))
    H2("Transformer trained from scratch")
    P("The second system is a Transformer encoder-decoder [@vaswani2017] trained from random "
      "initialisation. It is built from PyTorch [@paszke2019] encoder and decoder layers; the sinusoidal "
      "positional encoding, the causal and padding masks, a single embedding shared by the source "
      "input, the target input and the output layer, and greedy and beam-search decoding are our own "
      "code. Because Hinglish spelling varies, it uses one joint SentencePiece BPE vocabulary of 4,000 "
      "pieces for both languages, trained on the training split. The vocabulary is small because the data "
      "is small: a larger one would leave many pieces with few examples. We trained two settings with "
      "d_model 256, 4 attention heads, feed-forward size 1,024, dropout 0.3, label smoothing 0.1 and the "
      "warmup learning-rate schedule of the original paper: a 3-layer model "
      f"({tr_small['parameters'] / 1e6:.1f}M parameters) and a 6-layer model ({tr['parameters'] / 1e6:.1f}M "
      "parameters), for up to 40 epochs on a CPU. The 6-layer model is reported as the main result.")
    H2("Pretrained models, zero-shot and fine-tuned")
    P("The third family uses pretrained translation models. mBART-50 "
      "(`facebook/mbart-large-50-many-to-many-mmt`) is given the Hindi source-language tag and English as "
      "target. mT5-small (`google/mt5-small`) receives the prefix “translate Hinglish to English:”. "
      "For both we report the zero-shot score (no training) and the score after fine-tuning on the "
      "training split (mBART-50 for 3 epochs, mT5-small for 5 epochs), with learning rate 3e-5, batch size 8 "
      "with 4 gradient-accumulation steps, and beam search of size 4, on a Colab T4 GPU through the "
      "Hugging Face library [@wolf2020]. The Roman-script input is passed to the tokenizers unchanged.")
    H2("Evaluation protocol")
    P("All systems are scored by one function. It lowercases predictions and references and computes "
      "corpus BLEU and chrF with sacreBLEU. Lowercasing is necessary because the references are "
      "lowercased by our preprocessing while the pretrained models write capitalised English; without it "
      "they would be penalised only for casing. After a system is trained, its 100 lowest-scoring test "
      "translations (by sentence chrF) are saved, and 50 of them are tagged by hand with an error "
      "category for the error analysis.")
    lc, tc = lstm["config"], json.loads((ROOT / "configs" / "transformer_deep.json").read_text(encoding="utf-8"))
    mh, th = mb["hyperparameters"], mt["hyperparameters"]
    blocks.append(("table", dict(
        label="settings", caption="Main settings of the trained and fine-tuned systems",
        cols=["Setting", "LSTM + attention", "Transformer (6 layers)", "mBART-50", "mT5-small"],
        align="p1p2p3p4p6",
        rows=[
            ["Vocabulary", f"words seen at least {lc['min_freq']} times", f"joint BPE, {tc['vocab_size']:,} pieces",
             "pretrained subwords", "pretrained subwords"],
            ["Size", f"embedding {lc['emb_dim']}, encoder 2x{lc['enc_hid_dim']}, decoder {lc['dec_hid_dim']}",
             f"d_model {tc['d_model']}, {tc['nhead']} heads, feed-forward {tc['ff_dim']:,}, {tc['num_layers']}+{tc['num_layers']} layers",
             "pretrained, large", "pretrained, small"],
            ["Dropout / label smoothing", f"{lc['dropout']} / {lc['label_smoothing']}",
             f"{tc['dropout']} / {tc['label_smoothing']}", "model default / none", "model default / none"],
            ["Optimiser, learning rate", f"Adam, {lc['lr']}", f"Adam, warmup {tc['warmup']} steps",
             f"AdamW, {sci(mh['lr'])}", f"AdamW, {sci(th['lr'])}"],
            ["Batch size", str(lc["batch_size"]), str(tc["batch_size"]),
             f"{mh['batch_size']} × {mh['grad_accum']} accumulation", f"{th['batch_size']} × {th['grad_accum']} accumulation"],
            ["Epochs", f"up to {lc['epochs']}, early stopping (stopped at 15)", f"up to {tc['epochs']} (no early stop)",
             str(mh["epochs"]), str(th["epochs"])],
            ["Beam size", str(lc["beam_size"]), str(tc["beam_size"]), str(mh["beam"]), str(th["beam"])],
            ["Seed", str(lc["seed"]), str(tc["seed"]), str(mh["seed"]), str(th["seed"])]])))

    # ---------------------------------------------------------------- 5 Results
    H1("Results")
    P(f"[[tab:results]] and [[fig:comparison]] show the test scores of all systems on the {n_test:,} test "
      f"sentences. The fine-tuned mBART-50 is clearly best ({d2(b(mb))} BLEU, {d2(c(mb))} chrF). The "
      f"Transformer trained from scratch is second ({d2(b(tr))} BLEU), and it is above the zero-shot mBART-50 "
      f"({d2(b(mbz))}). Fine-tuned mT5-small ({d2(b(mt))}) and the LSTM ({d2(b(lstm))}) follow, and zero-shot "
      f"mT5-small is useless ({d2(b(mtz))} BLEU).")
    blocks.append(("table", dict(
        label="results", caption="Test results (1,345 sentences); higher is better",
        cols=["Rank", "System", "Training", "BLEU", "chrF"], align="llllrr"[:5] if False else "rllrr",
        rows=[[str(i + 1), nm, how, d2(b(s)), d2(c(s))] for i, (nm, how, s) in enumerate(sorted([
            ("mBART-50 fine-tuned", "pretrained + fine-tuned", mb),
            ("Transformer, 6 layers", "from scratch", tr),
            ("mBART-50 zero-shot", "pretrained, no training", mbz),
            ("mT5-small fine-tuned", "pretrained + fine-tuned", mt),
            ("LSTM + attention", "from scratch", lstm),
            ("mT5-small zero-shot", "pretrained, no training", mtz)], key=lambda x: -x[2]["bleu"]))])))
    blocks.append(("fig", dict(
        label="comparison", file="figures/comparison.png", width=0.85,
        caption="BLEU and chrF of all systems on the test set.")))
    H2("Effect of decoding and model size")
    P(f"Beam search improved over greedy decoding for both systems we trained from scratch ([[tab:decoding]]): "
      f"by {b(lstm) - lstm['greedy']['bleu']:.2f} BLEU for the LSTM and {b(tr) - tr['greedy']['bleu']:.2f} "
      f"for the 6-layer Transformer. The 3-layer Transformer ({d2(b(tr_small))} BLEU, {d2(c(tr_small))} chrF) "
      f"is practically as good as the 6-layer one ({d2(b(tr))} BLEU, {d2(c(tr))} chrF): the BLEU difference "
      f"of {b(tr) - b(tr_small):.2f} is within the noise of a single run, and the deeper model needed about "
      "twice the training time. Neither Transformer run stopped early; the validation loss was still "
      "falling slowly at epoch 40, so longer training might give a little more.")
    blocks.append(("table", dict(
        label="decoding", caption="Greedy decoding against beam search (test BLEU)",
        cols=["System", "Greedy", "Beam search", "Gain"], align="lrrr",
        rows=[[nm, d2(s["greedy"]["bleu"]), d2(b(s)), f"{b(s) - s['greedy']['bleu']:.2f}"]
              for nm, s in (("LSTM + attention (beam 5)", lstm), ("Transformer, 3 layers (beam 5)", tr_small),
                            ("Transformer, 6 layers (beam 5)", tr))])))
    blocks.append(("fig", dict(
        label="curves", file="figures/lstm_curves.png", width=0.98,
        caption="Training curves of the LSTM: the training loss keeps falling while the validation loss "
                "stays flat after about epoch 5 (left), and the validation BLEU rises slowly (right).")))

    # ---------------------------------------------------------------- 6 Error analysis
    H1("Error analysis")
    P("For each trained system, the 50 worst test translations (lowest sentence chrF) were tagged by hand "
      "with an error category. The categories were chosen by the person tagging each system, so the columns "
      "of [[tab:errors]] are only roughly comparable, and the table describes the worst sentences, not all "
      "errors. [[tab:examples]] shows real examples.")
    blocks.append(("table", dict(
        label="errors", caption="Hand-tagged error categories among the 50 worst test translations of each system",
        cols=["Error category"] + list(cats), align="l" + "r" * len(cats), rows=cat_rows)))
    blocks.append(("bullets_head", None))
    blocks.append(("ul", [
        "**mBART-50** writes fluent English. Its worst cases are mostly a wrong meaning for a colloquial "
        "phrase, or a noisy reference.",
        "**mT5-small** mostly drops words or repeats itself, often producing only a mention or a few "
        "characters.",
        "**The Transformer** almost never produces junk output, but when it does not know a Hindi word it "
        "often copies it into the English, because both languages share one BPE vocabulary.",
        f"**The LSTM** produces junk such as `) :p` in {cats['LSTM'].get('Junk output', 0)} of its 50 worst "
        "cases, which its author traced to emoticon-only training references.",
        "**Noisy references** are among the worst cases of every system (3 to 14 of 50). For example, the "
        "reference of `rt @sheikhimaan : news flash : iss baar bheja tha . jeet gaye .` is only `) <url>`."]))
    blocks.append(("table", dict(
        label="examples", caption="Examples of tagged errors (real rows from the error files)",
        cols=["Hinglish input", "Reference", "System", "System output", "Category"], align="p1p2p3p4p5",
        rows=ex)))

    # ---------------------------------------------------------------- 7 Discussion
    H1("Discussion and limitations")
    H2("Does pretraining help?")
    P(f"For mBART-50, yes. Fine-tuned mBART-50 is {b(mb) - b(tr):.2f} BLEU above the best system trained "
      f"from scratch, and fine-tuning added {b(mb) - b(mbz):.2f} BLEU over its zero-shot score. Even "
      f"zero-shot, mBART-50 ({d2(b(mbz))}) is {b(mbz) - b(lstm):.2f} BLEU above the LSTM. For mT5-small the answer "
      f"is no: zero-shot it is useless, and after fine-tuning ({d2(b(mt))}) it stays {b(tr) - b(mt):.2f} BLEU "
      "below the Transformer trained from scratch. Likely reasons are that mT5-small is a small "
      "general-purpose model that was not pretrained for translation, and that we fine-tuned it with a "
      "learning rate of 3e-5, lower than the 3e-4 that is commonly used for mT5, so it may be under-tuned. "
      "We did not test this. The honest summary is that a pretrained model built for translation helps a "
      "lot, while a small general-purpose model does not necessarily.")
    H2("A Transformer from scratch is competitive")
    P(f"With only about 10.8 thousand training pairs, the Transformer reached {d2(b(tr))} BLEU, above zero-shot "
      f"mBART-50. The BPE vocabulary is probably a large part of the reason: the LSTM, with word-level "
      f"vocabularies, loses many rare words to `<unk>`. The {b(tr) - b(lstm):.1f} BLEU gap between the LSTM "
      "and the Transformer is therefore a mixture of architecture and tokenisation; we did not train an LSTM "
      "with BPE to separate the two.")
    H2("Limitations")
    blocks.append(("ul", [
        "**Single runs.** Every system was trained once with seed 42. Differences of a few tenths of BLEU, such as "
        f"the {b(tr) - b(tr_small):.2f} BLEU between the two Transformers, are not meaningful. We computed no "
        "confidence intervals or significance tests, because only the 100 worst predictions of each system "
        "were saved, not all test predictions.",
        "**Noisy references.** PHINC translations are sometimes wrong or loosely aligned, which lowers all "
        "scores. Removing the emoticon-only test references would change the split again, so we kept them.",
        "**Not done.** IndicTrans2 and a Roman-to-Devanagari transliteration experiment were prepared but "
        "not run; mT5 was not tuned; we did not train on smaller fractions of the data to see where "
        "pretraining helps most, and we did not break the scores down by sentence length or amount of "
        "code-mixing.",
        "**Different taggers.** The errors of each system were tagged by a different person with their own "
        "categories.",
        "**One dataset.** All data is social media text from one corpus; the results may not carry over to other "
        "kinds of Hinglish."]))

    # ---------------------------------------------------------------- 8 Conclusion
    H1("Conclusion")
    P(f"On PHINC, a pretrained translation model fine-tuned on our small training set (mBART-50, {d2(b(mb))} "
      f"BLEU) is clearly better than a Transformer trained from scratch ({d2(b(tr))} BLEU), which is in turn far "
      f"better than an LSTM with attention ({d2(b(lstm))} BLEU). The result does not extend to every pretrained "
      f"model, since fine-tuned mT5-small ({d2(b(mt))} BLEU) is below the from-scratch Transformer. The most "
      "useful lessons of the project concerned data and evaluation: cleaning junk translations, cut tweets "
      "and HTML codes from the raw file, and scoring all systems in exactly the same way, changed which "
      "conclusions we could trust. Future work is to run IndicTrans2, to tune mT5, to repeat the runs with "
      "several seeds and test significance, and to measure where pretraining helps most by training on "
      "growing fractions of the data.")
    blocks.append(("h1_star", "Author contributions"))
    blocks.append(("table", dict(
        label="contrib", caption="Contributions of the authors", cols=["Author", "Contribution"], align="p1p2",
        rows=[
            ["Swati Kumari", "Repository, data pipeline (download, cleaning, fixed split, vocabulary helper), shared "
                             "BLEU/chrF evaluation, tests and continuous integration, comparison script, README and this report."],
            ["Pinky Rana", "LSTM with Bahdanau attention, its results and error analysis; code reviews that found "
                           "the lowercasing bug, the junk translations and the cut tweets in the data."],
            ["Vinay Kumar", "Fine-tuning code for mBART-50, mT5 and IndicTrans2, the Colab notebook, and the "
                            "fine-tuned and zero-shot results and error analysis for mBART-50 and mT5."],
            ["Ujjval Baijal", "Transformer trained from scratch, the comparison of two model sizes, its results "
                              "and error analysis."]])))
    blocks.append(("h1_star", "Code and data availability"))
    P("The code, the result files and the continuous-integration setup are in the project repository "
      "(github.com/Swati89923/Translate-Roman-script-code-mixed-Hinglish-into-English). PHINC is publicly "
      "available from Zenodo (doi:10.5281/zenodo.3605597); data files and model checkpoints are not "
      "committed. Running `python src/compare.py` regenerates the ranking, the chart and the error tables "
      "from the result files.")

    return dict(
        title="Translating code-mixed Hinglish into English: how much does pretraining help "
              "when data is scarce?",
        short_title="Hinglish to English machine translation",
        authors=["Swati Kumari", "Pinky Rana", "Vinay Kumar", "Ujjval Baijal"],
        affiliation="Group 8 (P08), Deep Learning course project, Sitare University, India",
        short_author="S. Kumari and others",
        received="Received 22 November 2026",
        abstract=(
            "Hinglish, Hindi written in the Latin alphabet and mixed with English, is common on social media "
            "but hard to translate because parallel data is scarce, spelling is not fixed and English words "
            "appear unchanged. We ask how much pretraining helps for translating Hinglish into English "
            "compared with training from scratch. On the PHINC corpus we fix a cleaned train/validation/test "
            "split, fixing junk translations, cut tweets and HTML codes found in the raw data, and compare "
            "four systems under one evaluation protocol: an LSTM with Bahdanau attention and a Transformer, "
            "both trained from scratch, and mBART-50 and mT5-small, both zero-shot and fine-tuned. "
            f"Fine-tuned mBART-50 is best ({d2(b(mb))} BLEU, {d2(c(mb))} chrF), followed by the Transformer "
            f"trained from scratch ({d2(b(tr))} BLEU), fine-tuned mT5-small ({d2(b(mt))}) and the LSTM "
            f"({d2(b(lstm))}). Pretraining therefore helped clearly for a model built for translation, but not for "
            "every pretrained model. A hand-tagged analysis of the worst translations shows different "
            "failure modes for each system, and noisy references account for a large share of them. All "
            "results are single runs on one corpus."),
        blocks=blocks,
        references=REFERENCES,
        pages_note=None,
    )
