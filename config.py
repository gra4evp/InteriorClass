MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


BASE_PROMPT_RU = """
Проанализируйте фотографию интерьера и классифицируйте его состояние. Выберите ТОЛЬКО ОДИН класс из предложенных ниже. В ответе укажите ТОЛЬКО метку класса (например "C0").

Детальные критерии классификации:

[A0] Без отделки:
- Бетонные/необработанные стены и пол
- Черновая стяжка без покрытия
- Открытые инженерные коммуникации (трубы, провода)
- Отсутствие радиаторов, дверей, сантехники
- Пример: строительная коробка без отделки

[A1] Требуется капитальный ремонт (сильный износ >70%):
- Трещины в стяжке и стенах
- Отваливающаяся штукатурка
- Поврежденные коммуникации (ржавые трубы, оголенные провода)
- Плесень и пятна от протечек
- Отсутствие финишных покрытий
- Пример: аварийное состояние помещения

[B0] White-box (под чистовую отделку):
- Стены и пол выровнены (штукатурка, стяжка)
- Разведена электрика (розетки, выключатели)
- Установлены базовые радиаторы (без декора)
- Нет финишных покрытий (ламинат, плитка, обои)
- Пример: помещение готово под финальную отделку

[B1] Требуется косметический ремонт (износ 30-50%):
- Инженерные системы исправны
- Потёртый или вздутый ламинат/линолеум/паркет/доска/плинтус/потолок
- Потёртая или отваливающаяся краска/плитка
- Устаревшая или повреждённая мебель/двери/окна
- Выцвевшие или отслаивающиеся обои
- Трещины в краске/штукатурке/плитке
- Устаревшая, но рабочая сантехника
- Изношенная ванна/душ/раковина/унитаз
- Пример: жилье после 5-7 лет эксплуатации

[C0] Хорошее состояние:
- Отделка не старше 5 лет
- Качественные материалы (ламинат 33 класса, моющиеся обои)
- Ровные стены без дефектов
- Современная сантехника
- Возможен тёплый пол
- Пример: квартира после недавнего ремонта

[C1] Отличное состояние:
- Инженерная доска/паркет
- Шпонированные панели
- Мульти-сплит система
- Фильтры для воды
- Скрытая разводка коммуникаций
- Дорогая (но не дизайнерская) мебель
- Пример: премиальный жилой комплекс

[D0] Дизайнерский ремонт (евроремонт):
- Реализованный дизайн-проект
- Натуральные материалы (камень, массив дерева)
- VRF-кондиционирование
- Умный дом (базовый уровень)
- Мебель из шпона/МДФ высокого класса
- Пример: студийный ремонт по проекту

[D1] Люкс (эксклюзив):
- Авторский дизайн-проект
- Элитные материалы (book-match камень, редкие породы дерева)
- Полная автоматизация (KNX, Crestron)
- Встроенная мебель (bespoke изготовление)
- Мультирум-аудиосистемы (7.1 и выше)
- Пример: резиденции премиум-класса

Отвечайте ТОЛЬКО меткой класса (A0, A1...D1) без пояснений.
"""


BASE_PROMPT_EN = """
Analyze the interior photo and classify its condition. Choose ONLY ONE class from the options below. Respond STRICTLY with ONLY the class label (e.g. "C0").

Detailed classification criteria:

[A0] No finish:
- Raw concrete walls/floors
- Unfinished screed
- Exposed utilities (pipes, wires)
- Missing radiators, doors, plumbing
- Example: construction shell without finishes

[A1] Needs major renovation (>70% wear):
- Cracks in screed/walls
- Peeling plaster
- Damaged utilities (rusty pipes, exposed wires)
- Mold and water stains
- No finish coatings
- Example: emergency condition premises

[B0] White-box (ready for finishing):
- Leveled walls/floors (plaster, screed)
- Electrical wiring installed (sockets, switches)
- Basic radiators (no covers)
- No final coatings (laminate, tiles, wallpaper)
- Example: space ready for final decoration

[B1] Needs cosmetic repairs (30-50% wear):
- Functional utilities
- Worn laminate/linoleum
- Peeling wallpaper
- Paint cracks
- Outdated but working plumbing
- Example: 5-7 year old housing

[C0] Good condition:
- Finishes <5 years old
- Quality materials (33-class laminate, washable wallpaper)
- Smooth defect-free walls
- Modern plumbing
- Possible underfloor heating
- Example: recently renovated apartment

[C1] Excellent condition:
- Engineered wood/parquet
- Veneer wall panels
- Multi-split AC system
- Water filtration
- Hidden utilities
- High-end (non-designer) furniture
- Example: premium residential complex

[D0] Design renovation:
- Executed design project
- Natural materials (stone, solid wood)
- VRF air conditioning
- Smart home (basic level)
- High-grade veneer/MDF furniture
- Example: studio-designed renovation

[D1] Luxury (exclusive):
- Custom design project
- Elite materials (book-match stone, rare woods)
- Full automation (KNX, Crestron)
- Bespoke built-in furniture
- Multi-room audio (7.1+)
- Example: premium class residences

Respond ONLY with the class label (A0, A1...D1) without explanations.
"""
