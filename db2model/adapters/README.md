# Адаптеры (веса ветки SFT/LoRA)

Сюда кладутся обученные LoRA-адаптеры — это и есть «знание о БД в весах»,
требуемый DoD недели 6 артефакт «веса/адаптеры».

Адаптеры обучаются на Kaggle (базы из Kaggle не видно, туда едет только датасет).
После прогона `kaggle_train_lora.ipynb` забери из выхода ноутбука
`adapter_synth347.zip` и распакуй сюда:

```
db2model/adapters/
  synth347/
    adapter_model.safetensors
    adapter_config.json
    adapter_card.json      # base_model, seed, LoRA-конфиг, хеш данных — провенанс
    tokenizer files...
```

`adapter_card.json` привязывает веса к данным (`manifest_sha1`) и гиперпараметрам,
чтобы прогон воспроизводился. Базовая модель — `Qwen/Qwen2.5-Coder-3B-Instruct`,
адаптер грузится поверх неё через `peft`.

Сами веса (~70 МБ) в git не коммитятся (см. `.gitignore`) — это бинарный артефакт,
он передаётся как output Kaggle-версии или релиз-ассет. В репозитории версионируется
только `adapter_card.json` каждого адаптера (провенанс).
