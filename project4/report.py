import os
import textwrap

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
REPORT_PATH = os.path.join(ARTIFACTS_DIR, 'report.pdf')

PAGE_SIZE = (8.5, 11)
WRAP_WIDTH = 92
HEADING_WRAP_WIDTH = 75  # headings render at a larger fontsize, so need a narrower wrap
LINE_HEIGHT = 0.021


def _wrapped_lines(paragraph, width=WRAP_WIDTH):
    if not paragraph:
        return ['']
    return textwrap.fill(paragraph, width=width).split('\n')


def _footer(fig, page_number, total_pages):
    fig.text(0.5, 0.03, f"Page {page_number} of {total_pages}", fontsize=8, ha='center', color='#888888')


def _text_page(pdf, heading, paragraphs, page_number, total_pages):
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.text(0.08, 0.95, heading, fontsize=18, weight='bold', va='top')
    fig.text(0.08, 0.915, '_' * 70, fontsize=10, va='top', color='#999999')

    y = 0.87
    for para in paragraphs:
        if para.startswith('## '):
            lines = _wrapped_lines(para[3:], width=HEADING_WRAP_WIDTH)
            fig.text(0.08, y, '\n'.join(lines), fontsize=12.5, weight='bold', va='top', linespacing=1.3)
            y -= LINE_HEIGHT * (len(lines) * 1.3 + 1.8)
            continue
        lines = _wrapped_lines(para)
        fig.text(0.08, y, '\n'.join(lines), fontsize=10.5, va='top', ha='left', linespacing=1.5)
        y -= LINE_HEIGHT * (len(lines) * 1.5 + 1.5)

    _footer(fig, page_number, total_pages)
    pdf.savefig(fig)
    plt.close(fig)


def _title_page(pdf):
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.text(0.5, 0.68, "Preference Elicitation", ha='center', fontsize=24, weight='bold')
    fig.text(0.5, 0.62, "Human-Centric Artificial Intelligence -- Project 4", ha='center', fontsize=14)
    fig.text(0.5, 0.55, "Comparing pairwise comparison vs. top-10 ranking for movie preference elicitation",
              ha='center', fontsize=11, style='italic')
    pdf.savefig(fig)
    plt.close(fig)


def _render_sigmoid_plot():
    path = os.path.join(ARTIFACTS_DIR, 'bradley_terry_curve.png')
    diffs = np.linspace(-6, 6, 200)
    probs = 1 / (1 + np.exp(-diffs))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(diffs, probs, color='#1f77b4')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel(r'Utility difference $w^Tx_i - w^Tx_j$')
    ax.set_ylabel(r'$P(i \succ j)$')
    ax.set_title('Bradley-Terry pairwise choice probability (Plackett-Luce at n=2)')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def build_report():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    sigmoid_path = _render_sigmoid_plot()
    total_pages = 8

    with PdfPages(REPORT_PATH) as pdf:
        _title_page(pdf)

        _text_page(pdf, "1. Feature Representation (Task 1)", [
            "The IMDB 5000 Movie Dataset (~5,000 movies, 28 raw attributes, no user ratings) "
            "is featurized into a fixed-length numeric vector x per movie, so that a linear "
            "utility U(x) = w^Tx can express a user's taste as a weighted combination of "
            "movie attributes.",

            "## Numeric attributes (log-transformed where skewed, median-imputed, then "
            "standardized)",
            "duration, title_year, imdb_score, num_voted_users, budget, gross, "
            "director_facebook_likes, actor_1_facebook_likes, cast_total_facebook_likes, "
            "movie_facebook_likes. The popularity/likes and financial figures are heavily "
            "right-skewed (a handful of blockbusters dwarf everything else), so they are "
            "log1p-transformed before standardizing -- otherwise a few outliers would "
            "dominate the feature scale and the utility function.",

            "## Categorical attributes",
            "Genres are multi-label (e.g. 'Action|Adventure|Sci-Fi'), so the 15 most "
            "frequent genres are each turned into a binary indicator column. Content rating "
            "(R, PG-13, PG, ...) is one-hot encoded, with a shared 'Other' bucket for rare "
            "ratings.",
        ], 1, total_pages)

        _text_page(pdf, "1. Feature Representation (continued)", [
            "## Deliberately excluded attributes",
            "Director name, actor names, plot keywords, movie title, and the IMDB link are "
            "excluded as features (though director/actor NAMES are dropped, their "
            "popularity -- facebook_likes -- is kept, above): one-hot encoding thousands of "
            "distinct people would almost never overlap between two randomly-shown movies "
            "and would not generalize to a new user's preferences over unseen people. "
            "Language, country, and color are excluded as near-constant in this dataset "
            "(over 93% English, 75% USA, 95% color) and therefore carry little information "
            "for distinguishing preferences.",

            "## Why include imdb_score",
            "This is a deliberate, debatable choice: imdb_score is itself an aggregate "
            "preference signal (other viewers' ratings), not a neutral content attribute, "
            "so including it lets the fitted w partially reflect 'agrees with the crowd' "
            "rather than purely idiosyncratic personal taste. It is kept because (a) many "
            "real recommender systems legitimately use aggregate quality as one input "
            "signal among several, and (b) a user's personal utility function may "
            "reasonably correlate with, without being identical to, the crowd's; this "
            "trade-off is worth stating explicitly rather than hiding.",

            "All movies missing genres or title_year (attributes that can't be reasonably "
            "imputed) are dropped, leaving 4,935 usable movies; all other missing numeric "
            "values are median-imputed rather than dropping the row, to keep the sampling "
            "pool as large as possible.",
        ], 2, total_pages)

        _text_page(pdf, "2. Extending Bradley-Terry to Rankings (Task 2)", [
            "The standard Bradley-Terry model gives the probability that item i is "
            "preferred to item j as a logistic function of their utility difference: "
            "P(i > j) = sigmoid(U(x_i) - U(x_j)) = 1 / (1 + exp(-(w^Tx_i - w^Tx_j))). "
            "Design 2 asks participants to produce a full ranking i1 > i2 > ... > in of n "
            "items rather than a single pairwise choice, which the standard model does not "
            "cover.",

            "## Proposed extension: the Plackett-Luce model",
            "The natural generalization treats a full ranking as a sequence of nested "
            "choices: first choose the most preferred item out of all n, then the most "
            "preferred out of the remaining n-1, and so on, at each step applying Luce's "
            "choice axiom (probability of choosing an item is proportional to "
            "exp(utility)):",

            "P(i1 > i2 > ... > in | w)  =  product over k=1..n-1 of "
            "[ exp(U(i_k)) / sum_{j=k}^{n} exp(U(i_j)) ]",

            "## Why this is the right generalization, not an arbitrary one",
            "First, it reduces EXACTLY to standard Bradley-Terry when n=2: the product has "
            "a single term, exp(U(i1)) / (exp(U(i1)) + exp(U(i2))), which is algebraically "
            "identical to the sigmoid form above. This means a pairwise comparison from "
            "Design 1 is simply a length-2 ranking under this model -- the same likelihood "
            "function and the same fitting code serve both designs, rather than needing two "
            "separate, potentially inconsistent models. Second, the negative log-likelihood "
            "is convex in w (each term is a log-sum-exp, itself convex, minus a linear "
            "term), so maximum-likelihood fitting is a well-behaved convex optimization "
            "problem rather than requiring heuristics or risking bad local optima. Third, "
            "the 'sequentially remove the winner and recurse' structure directly mirrors "
            "how a person plausibly constructs the ranking in their own head (identify the "
            "best, then the best of what's left, ...), which is a natural process "
            "assumption for a ranking-elicitation task.",

            "## Regularization",
            "In practice, a participant gives only a handful of responses (15 pairwise "
            "comparisons, or 4 rankings of 10 movies) relative to the ~32-dimensional "
            "feature space, so plain maximum-likelihood estimation is ill-posed (many w "
            "vectors would fit the small number of observed comparisons equally well, or "
            "the optimizer could drive weights to infinity if some feature happens to "
            "perfectly separate the few observed choices). The implementation therefore "
            "fits the MAP estimate under a Gaussian(0, 1/reg) prior on w -- equivalent to "
            "adding an L2 penalty 0.5*reg*||w||^2 to the negative log-likelihood -- which "
            "keeps the estimate finite and well-conditioned with few responses.",
        ], 3, total_pages)

        fig = plt.figure(figsize=PAGE_SIZE)
        fig.text(0.08, 0.95, "2. Extending Bradley-Terry to Rankings (continued)", fontsize=16, weight='bold', va='top')
        fig.text(0.08, 0.915, '_' * 70, fontsize=10, va='top', color='#999999')
        ax = fig.add_axes([0.15, 0.55, 0.7, 0.32])
        img = plt.imread(sigmoid_path)
        ax.imshow(img)
        ax.axis('off')
        caption = _wrapped_lines(
            "The pairwise case (n=2) of the Plackett-Luce model is exactly the classic "
            "Bradley-Terry sigmoid curve: the probability of preferring item i grows "
            "smoothly with the utility difference w^Tx_i - w^Tx_j, saturating toward 0 or 1 "
            "for large differences and equal to 0.5 when the two items have equal utility."
        )
        fig.text(0.1, 0.45, '\n'.join(caption), fontsize=9.5, style='italic', va='top')
        _footer(fig, 4, total_pages)
        pdf.savefig(fig)
        plt.close(fig)

        _text_page(pdf, "3. User Study Design (Task 3)", [
            "This section describes a complete protocol for comparing Design 1 (pairwise) "
            "against Design 2 (top-10 ranking) as preference-elicitation interfaces. As "
            "instructed, the study is designed but not run.",

            "## Hypotheses",
            "H1 (efficiency): a 10-item ranking implies up to C(10,2)=45 pairwise "
            "preference constraints but takes far less than 45x as long to produce, so for "
            "a matched amount of participant TIME, Design 2 will yield a preference "
            "estimate with higher held-out prediction accuracy than Design 1. "
            "H2 (usability trade-off): Design 1 will be rated as less cognitively "
            "demanding and more subjectively preferred than Design 2, i.e. there is an "
            "efficiency-vs-burden trade-off between the two interfaces worth quantifying, "
            "not a simple 'one design is strictly better' outcome.",

            "## Design",
            "Within-subjects, counterbalanced: every participant completes BOTH designs "
            "(so individual differences in movie taste/engagement cancel out in the "
            "comparison), with presentation order randomized across participants (half "
            "complete Design 1 then Design 2, half the reverse) to control for order and "
            "learning effects. Each condition block ends with the same fixed set of "
            "validation pairwise questions (movies never used for fitting, identical "
            "across both conditions and all participants) so that held-out prediction "
            "accuracy is directly comparable between designs and across participants.",
        ], 5, total_pages)

        _text_page(pdf, "3. User Study Design (continued)", [
            "## Dependent variables",
            "(1) Validation accuracy: fraction of held-out validation pairs correctly "
            "predicted by the condition's fitted preference vector. (2) Interaction time: "
            "wall-clock seconds spent completing the elicitation block, measured "
            "client-side. (3) Subjective ease: a short post-block Likert-scale question "
            "(1-7, 'How easy was this task?') plus a single forced-choice 'which did you "
            "prefer overall?' at the very end.",

            "## Recruitment and sample size",
            "Participants recruited via a crowdsourcing platform (e.g. Prolific) or a "
            "university participant pool, screened to exclude people who report rarely "
            "watching movies (their preferences would be noisy/meaningless for this task). "
            "For a within-subjects paired comparison (paired t-test, alpha=0.05, power=0.8, "
            "assumed medium effect size d=0.5), the standard sample-size formula "
            "n = ((z_alpha/2 + z_beta) / d)^2 gives approximately n=34 pairs of "
            "observations; recruiting roughly 40-50 participants leaves margin for "
            "exclusions/dropouts.",
        ], 6, total_pages)

        _text_page(pdf, "3. User Study Design (continued, 2)", [
            "## Procedure",
            "(1) Informed consent and a brief movie-familiarity screening question. "
            "(2) Random assignment to presentation order (Design 1 first vs. Design 2 "
            "first). (3) First condition block: instructions, the elicitation rounds, then "
            "the fixed validation questions, then the subjective-ease rating. (4) Short "
            "distractor/break task to reduce carryover fatigue. (5) Second condition block, "
            "identical structure. (6) Final forced-choice overall preference question and "
            "debrief.",

            "## Ethics",
            "No personally identifying information is collected beyond a broad age "
            "range; participation is voluntary with an explicit right to withdraw at any "
            "point without penalty; movie-preference data is anonymized and reported only "
            "in aggregate; the protocol would be submitted for institutional ethics "
            "(IRB) approval before running with real participants.",

            "## Analysis plan",
            "Paired t-test (or Wilcoxon signed-rank if the accuracy differences are not "
            "approximately normal, checked via a Shapiro-Wilk test) comparing validation "
            "accuracy between conditions; a paired test on interaction time; McNemar's "
            "test or a simple proportion test on the final forced-choice preference. Order "
            "(which design was first) is included as a between-subjects factor to check "
            "for and control any learning/fatigue effect before pooling.",
        ], 7, total_pages)

        _text_page(pdf, "4. Implementation Notes (Task 4)", [
            "The interface implementing both designs is available directly from the "
            "project landing page. 'Start the study' randomly assigns a visitor to Design "
            "1 or Design 2, mirroring how a real deployment would randomize condition "
            "assignment for a between-subjects or counterbalanced within-subjects rollout; "
            "explicit 'Preview Design 1' / 'Preview Design 2' links are also provided so "
            "both interfaces can be inspected directly without relying on random chance.",

            "Design 1 presents 15 rounds of two randomly sampled movies with a single "
            "'I'd rather watch this' choice per round. Design 2 presents 4 rounds of ten "
            "randomly sampled movies each, ranked via a numbered dropdown per movie "
            "(validated server-side as a genuine permutation of 1..10) rather than "
            "drag-and-drop, to stay within this project's plain HTML/CSS/vanilla-JS "
            "approach. Both designs then present the same 3 validation pairwise questions, "
            "fit the shared Plackett-Luce/Bradley-Terry preference vector described in "
            "Section 2, and show the participant their top-weighted features and their "
            "top-5 recommended (previously-unseen) movies from the dataset -- a concrete, "
            "immediate payoff that also doubles as a live demonstration of the validation "
            "methodology proposed in Section 3.",

            "All elicitation state (responses collected so far, which movies have already "
            "been shown, current phase) is held in the Django session rather than a "
            "database, since no persistent cross-session record is required for a design "
            "demonstration -- consistent with how Project 3's optional interactive "
            "component is implemented in this same codebase.",
        ], 8, total_pages)

    return REPORT_PATH


if __name__ == '__main__':
    path = build_report()
    print(f"Report written to {path}")
