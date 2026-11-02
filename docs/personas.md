# Personas

Every simulated user in NeMo User Sim is a person sampled from
[Nemotron-Personas](https://huggingface.co/collections/nvidia/nemotron-personas),
a family of synthetic-population datasets grounded in census data. A persona
supplies the demographics, the personality traits and the language the
simulated user brings to a conversation, which is what makes a run something
other than one model's idea of a generic user.

NeMo User Sim samples an **extended version of these datasets distributed through
NGC**, which adds the OCEAN personality traits, names and a number of further
fields beyond those in the Hugging Face collection. Those extra fields are
what drive interaction style, so the NGC version is what the behavioral
features described below depend on.

## What a persona carries

Each row is one synthetic individual: name, age, sex, location, occupation,
education, and a written description in the region's own language. Alongside
those are the five OCEAN personality traits (openness, conscientiousness,
extraversion, agreeableness, neuroticism), which NeMo User Sim maps to an
interaction style. That mapping is what produces an impatient user who
withholds detail rather than a cooperative one who volunteers everything, and
it is the axis reports stratify on.

The people are synthetic. They are generated to reproduce the joint
distribution of real census attributes, not copied from it, so no row
corresponds to a real person.

## Getting the datasets

The datasets are not bundled, and they are large: a single locale runs from a
few hundred megabytes to several gigabytes. You download each locale you need
once, from the
[NGC catalog](https://catalog.ngc.nvidia.com/resources?query=nemotron-personas),
with Data Designer's `data-designer download personas` command. It stores them
in `~/.data-designer/managed-assets/datasets/`, where runs read them.

There is one resource per locale, named `nemotron-personas-dataset-<locale>`,
for example
[`nemotron-personas-dataset-en_us`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemotron-personas/resources/nemotron-personas-dataset-en_us).
`data-designer download personas --list` shows which locales exist and how
large each one is.

The download goes through the NGC CLI, so you need:

1. A free NGC account and an API key from
   [org.ngc.nvidia.com/account/api-keys](https://org.ngc.nvidia.com/account/api-keys).
2. The NGC CLI. On Linux, `usersim setup-ngc` installs it into the active
   virtual environment. On other platforms, such as macOS, use
   [NVIDIA's installer](https://org.ngc.nvidia.com/setup/installers/cli).
3. A CLI configuration: `ngc config set` stores your key and org, once per
   machine.

```bash
export NGC_CLI_API_KEY="your-ngc-key"
usersim setup-ngc                                # Linux; elsewhere, NVIDIA's installer
ngc config set                                   # enter your key, pick your org
data-designer download personas --locale en_US   # repeat --locale for more
```

`setup-ngc` is idempotent and verifies the download by sha256. It puts the CLI
inside the virtual environment rather than on your system path, so it needs no
administrator access.

**Which commands need this.** `usersim panel` and `usersim simulate` sample
personas, so both need the datasets; `simulate --inputs` runs stored rows and
doesn't. `usersim smoke` and any `--dry-run` resolve their full plan without
touching them, so you can confirm an install is sound before setting up NGC at
all.

## Licensing

The datasets carry their own license, separate from this project's Apache-2.0.

The extended datasets on [NGC](https://catalog.ngc.nvidia.com/resources?query=nemotron-personas)
are open, under the
[NVIDIA Dataset License Agreement](https://assets.ngc.nvidia.com/products/api-catalog/legal/Personas-Dataset-NVIDIA-Training-Dataset-License-Agreement-2025-11-26.pdf):
use, reproduction, modification and derivative works for training and
developing AI solutions, with no non-commercial restriction. It adds an
**AI ethics clause**, consistent with NVIDIA's Trustworthy AI terms.

The Hugging Face collection is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Exported personas

`usersim select` writes training data derived from runs. Its default export
schema carries the derived and aggregate persona fields, such as the age bin
and the interaction style, and leaves out two sets of columns unless you ask
for them: the fields copied straight from the source dataset (including the
whole sampled person), and the protected attributes in
`persona_religion_language_context`. Add them back with `--schema full`, the
`persona_verbatim` and `persona_protected` group tokens, or by column name.
