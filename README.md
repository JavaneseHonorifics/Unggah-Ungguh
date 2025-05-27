# Javanese Honorifics (Unggah-Ungguh v1.0)

The Javanese language, spoken by over 98 million people, features a distinctive honorific system known as Unggah-Ungguh Basa. We present UNGGAH-UNGGUH, a carefully curated dataset designed to encapsulate the nuances of Unggah-Ungguh Basa, the Javanese speech etiquette framework that dictates the choice of words and phrases based on social hierarchy and context.

- **Paper:** https://arxiv.org/pdf/2502.20864
- **Models and Dataset:** https://huggingface.co/JavaneseHonorifics
- **Venue:** ACL 2025 Main Conference

## Dataset Description

Current version of the dataset (v1.0) covers ~4k paralel sentences for translation category and 160 instances for conversation category.

- **Funded by:** Lembaga Pengelola Dana Pendidikan (LPDP)
- **Language:** Javanese
- **Licence:** CC-BY-NC 4.0

<!-- ## Dataset Structure

Unggah-Ungguh present 2 categories: Conversation and Translation.
The translation dataset (~4K instances):

- index: Unique ID for each instance
- label: Honorific level of the Javanese sentence (0: Ngoko, 1: Ngoko Alus, 2: Krama, 3: Krama Alus)
- javanese sentence: A sentence in Javanese with a specific honorific level
- group: Group ID for related sentence sets
- indonesian sentence: Translation in Indonesian
- english sentence: Translation in English

The conversation dataset (160 instances):

- index: Unique ID for each instance
- role a: Social role of Speaker A
- role b: Social role of Speaker B
- context: Description of the conversation context
- a utterance: (EXAMPLE) Appropriate utterance by Speaker A
- a utterance category: Honorific level of Speaker A’s utterance
- b utterance: (EXAMPLE) Appropriate utterance by Speaker B
- b utterance category: Honorific level of Speaker B’s utterance -->

## Citation
```
@article{farhansyah2025language,
  title={Do Language Models Understand Honorific Systems in Javanese?},
  author={Farhansyah, Mohammad Rifqi and Darmawan, Iwan and Kusumawardhana, Adryan and Winata, Genta Indra and Aji, Alham Fikri and Wijaya, Derry Tanti},
  journal={arXiv preprint arXiv:2502.20864},
  year={2025}
}
```

## Dataset Card Authors

- M. Rifqi Farhansyah ([@rifqifarhansyah](https://github.com/rifqifarhansyah))
- Iwan Darmawan
- Adryan Kusumawardhana
- Genta Indra Winata ([@gentaiscool](https://github.com/gentaiscool))
- Alham Fikri Aji ([@afaji](https://github.com/afaji))
- Derry Tanti Wijaya ([@derrywijaya](https://github.com/derrywijaya))
