import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from django.conf import settings
from django.shortcuts import render

from . import ml


TREE_PLOT_PATH = os.path.join(settings.MEDIA_ROOT, 'p2_tree.png')
PDP_PLOT_PATH = os.path.join(settings.MEDIA_ROOT, 'p2_pdp.png')
ALE_PLOT_PATH = os.path.join(settings.MEDIA_ROOT, 'p2_ale.png')
TRADEOFF_PLOT_PATH = os.path.join(settings.MEDIA_ROOT, 'p2_tradeoff.png')

LAMBDA_MIN = 0.0
LAMBDA_MAX = 0.3
LAMBDA_STEP = 0.005
LAMBDA_DEFAULT = 0.02


def _row_choices(df):
    return [
        {
            'index': i,
            'label': f"#{i} — {row['species']}, {row['island']}, "
                     f"bill {row['bill_length_mm']:.0f}mm, flipper {row['flipper_length_mm']:.0f}mm",
        }
        for i, row in df.iterrows()
    ]


def _render_tree_plot(model):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(
        model,
        feature_names=ml.ENCODED_COLUMNS,
        class_names=list(model.classes_),
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(TREE_PLOT_PATH)
    plt.close(fig)


def _coef_table(pipeline):
    clf = pipeline.named_steps['clf']
    classes = list(clf.classes_)
    rows = []
    for i, col in enumerate(ml.ENCODED_COLUMNS):
        values = [round(float(clf.coef_[c, i]), 3) for c in range(len(classes))]
        if any(abs(v) > ml.COEF_EPS for v in values):
            rows.append({'feature': col, 'values': values})
    rows.sort(key=lambda r: max(abs(v) for v in r['values']), reverse=True)
    return {'classes': classes, 'rows': rows}


def _render_curves_plot(path, x_values, curves, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    for species in ml.SPECIES:
        ax.plot(x_values, curves[species], label=species, color=ml.SPECIES_COLORS[species])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_tradeoff_plot(family, selected, omega_label, lam):
    family_sorted = sorted(family, key=lambda r: r['omega'])
    omegas = [r['omega'] for r in family_sorted]
    accs = [r['acc'] for r in family_sorted]
    objectives = [r['acc'] - lam * r['omega'] for r in family_sorted]

    fig, ax = plt.subplots()
    ax.plot(omegas, accs, marker='o', color='#1f77b4', label='Test accuracy')
    ax.scatter(
        [selected['omega']], [selected['acc']],
        color='red', s=120, zorder=5, label='Selected model',
    )
    ax.set_xlabel(omega_label)
    ax.set_ylabel('Test accuracy')

    ax2 = ax.twinx()
    ax2.plot(omegas, objectives, color='gray', linestyle='--', alpha=0.5, label='acc − λ·Ω (objective)')
    ax2.set_ylabel('acc − λ·Ω')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)

    ax.set_title('Accuracy vs. complexity')
    fig.tight_layout()
    fig.savefig(TRADEOFF_PLOT_PATH)
    plt.close(fig)


def _model_info(model_type, family, selected, lam, ds):
    omega_label = 'Number of leaves' if model_type == 'tree' else 'Nonzero coefficients'
    info = {
        'model_type': model_type,
        'acc': round(selected['acc'], 3),
        'omega': selected['omega'],
        'omega_label': omega_label,
    }
    if model_type == 'tree':
        _render_tree_plot(selected['model'])
        info['tree_plot_url'] = f"{settings.MEDIA_URL}p2_tree.png?v={int(os.path.getmtime(TREE_PLOT_PATH))}"
    else:
        info['coef_table'] = _coef_table(selected['model'])

    # Regenerated on every request (unlike the cached family/selected model itself),
    # since which point is marked as "selected" depends on the current lambda.
    _render_tradeoff_plot(family, selected, omega_label, lam)
    info['tradeoff_plot_url'] = (
        f"{settings.MEDIA_URL}p2_tradeoff.png?v={int(os.path.getmtime(TRADEOFF_PLOT_PATH))}"
    )
    return info


def index(request):
    ds = ml.get_dataset()

    model_type = request.POST.get('model_type', 'tree')
    lam = float(request.POST.get('lam', LAMBDA_DEFAULT))
    row_index = int(request.POST.get('row_index', 0))
    default_target = next(s for s in ml.SPECIES if s != ds.df.iloc[row_index]['species'])
    target_label = request.POST.get('target_label', default_target)
    feature = request.POST.get('feature', ml.NUMERIC_FEATURES[0])

    context = {
        'species_options': ml.SPECIES,
        'feature_options': [{'key': f, 'label': ml.FEATURE_LABELS[f]} for f in ml.NUMERIC_FEATURES],
        'lambda_min': LAMBDA_MIN,
        'lambda_max': LAMBDA_MAX,
        'lambda_step': LAMBDA_STEP,
        'selected_model_type': model_type,
        'selected_lambda': lam,
        'selected_feature': feature,
        'selected_target': target_label,
        'selected_row': row_index,
        'row_options': _row_choices(ds.df),
    }

    # Every section is recomputed on every submit, using whichever model/lambda/inputs
    # are currently set in the (single, shared) form -- so the three regions always
    # stay consistent with each other rather than only the just-clicked one updating.
    family = ml.get_family(model_type, ds)
    selected = ml.select_by_lambda(family, lam)
    context['model_info'] = _model_info(model_type, family, selected, lam, ds)

    x_row = ds.df.iloc[row_index]
    context['x_row'] = x_row.to_dict()
    if x_row['species'] == target_label:
        context['counterfactual_error'] = (
            f"Row #{row_index} is already classified as {target_label}. "
            "Pick a different target label."
        )
    else:
        cf = ml.generate_counterfactuals(selected['model'], ds, x_row, target_label, k=5)
        if cf is None:
            context['counterfactual_error'] = (
                "No counterfactuals found for this example/target within the sampling budget."
            )
        else:
            display_rows = []
            for rec in cf.to_dict('records'):
                cells = []
                for col in ml.FEATURE_COLUMNS:
                    v = rec[col]
                    if col in ml.NUMERIC_FEATURES:
                        v = round(float(v), 1)
                        diff = v - float(x_row[col])
                        changed = abs(diff) > 0.01
                        delta = f"{diff:+.1f}" if changed else None
                    else:
                        changed = v != x_row[col]
                        delta = None
                    cells.append({'value': v, 'changed': changed, 'delta': delta})
                display_rows.append({'cells': cells, 'distance': round(rec['distance'], 3)})
            context['counterfactual_columns'] = ml.FEATURE_COLUMNS
            context['counterfactual_rows'] = display_rows

    grid = ml.pdp_grid(ds, feature)
    pdp_curves = ml.compute_pdp(selected['model'], ds.X_train, feature, grid)
    _render_curves_plot(
        PDP_PLOT_PATH, grid, pdp_curves,
        f"PDP — {ml.FEATURE_LABELS[feature]}", ml.FEATURE_LABELS[feature], "Predicted probability",
    )
    context['pdp_plot_url'] = f"{settings.MEDIA_URL}p2_pdp.png?v={int(os.path.getmtime(PDP_PLOT_PATH))}"

    bin_centers, ale_curves = ml.compute_ale(model_type, selected['model'], ds.X_train, feature)
    _render_curves_plot(
        ALE_PLOT_PATH, bin_centers, ale_curves,
        f"ALE — {ml.FEATURE_LABELS[feature]}", ml.FEATURE_LABELS[feature], "Accumulated local effect",
    )
    context['ale_plot_url'] = f"{settings.MEDIA_URL}p2_ale.png?v={int(os.path.getmtime(ALE_PLOT_PATH))}"

    return render(request, 'project2/index.html', context)
