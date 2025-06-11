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


TRASH_FILTER_PROMPT_RU = """
Ты эксперт по анализу фотографий недвижимости. Определи, является ли изображение мусором (непригодным для оценки интерьера). Отвечай ТОЛЬКО "1" (мусор) или "0" (интерьер).

Категории мусора (всегда отвечай "1"):
1. Фасады зданий (внешний вид домов)
2. Подъезды/лестничные площадки
3. Дворы/парковки/улицы
4. Чертежи/планировки (схемы сверху)
5. Фото документов/скриншоты
6. Люди/животные крупным планом
7. Пустые белые изображения (артефакты загрузки)
8. Нефокусные/засвеченные кадры
9. Части мебели без контекста комнаты
10. Окна/двери крупным планом без вида помещения

Если НИ ОДИН критерий не подходит - отвечай "0". Не объясняй.
"""


A0_ONLY_FILTER_PROMPT_RU = """
Проанализируй фотографию чернового состояния помещения. Твоя задача помочь отфильтровать изображения в датасете, класс [A0] от всего остального [UNKNOWN].
В ответе укажите ТОЛЬКО метку класса [A0]/[UNKNOWN] и его уверенность (например: "A0 8").
[A0] Без отделки (черновое состояние):
Отличительные черты:
- Бетонные/кирпичные/необработанные стены без шпаклевки, штукатурки или декоративной отделки
- Пол из черновой стяжки или голого бетона, без какого-либо покрытия (плитки, ламината и т.д.)
- Потолок в виде бетонной плиты перекрытия без финишной отделки, возможно наличие открытых коммуникаций, проводки, монтажных петель.
- Открытые инженерные коммуникации (трубы, провода, кабель-каналы)
Исключающие признаки (если есть ХОТЯ БЫ ОДИН – это НЕ A0):
- Наличие любой мебели, кухонной гарнитуры, техники. радиаторов, дверей, сантехники
- Наличие межкомнатных дверей, подоконников.
- Наличие даже частичной отделки (например, оштукатурены только стены или залита самовыравнивающаяся смесь на пол).

ГЛАВНЫЙ КРИТЕРИЙ [A0]: Полное отсутствие финишных отделочных слоев на всех поверхностях.
ПРИМЕР: Строительная коробка без отделки.

ЛЮБОЕ ДРУГОЕ ИЗОБРАЖЕНИЕ не подходящее под описание классифицируй как [UNKNOWN]. Если есть исключающие признаки классифицируй как [UNKNOWN].
Отвечайте ТОЛЬКО меткой класса [A0]/[UNKNOWN] и его уверенностью, без пояснений.
"""


D0_ONLY_FILTER_PROMPT_RU = """
Проанализируй фотографию помещения. Твоя задача — отфильтровать изображения класса [D0] (Эксклюзив / Luxury) от всего остального [UNKNOWN].
В ответе укажи ТОЛЬКО метку класса [D0]/[UNKNOWN] и уровень уверенности (например: "D0 9").

[D0] Дизайнерский ремонт (евроремонт):
ОТЛИЧИТЕЛЬНЫЕ ЧЕРТЫ:
1) Стены:
  - Декоративная штукатурка (венецианская, микроцемент), краска с фактурой.
  - Натуральные материалы: каменные панели (без book-match), деревянные панели (шпон, массив).
  - Возможен умеренный декор: молдинги, ниши с подсветкой.
2) Пол:
  - Натуральные материалы: паркетная доска (дуб, ясень), керамогранит под камень.
  - Бесшовные покрытия: наливные полы с декоративным эффектом.
3) Потолок:
  - Подвесные конструкции (гипсокартон, реечные системы) с точечной/ленточной подсветкой.
  - Минималистичные многоуровневые потолки (без сложного дизайна, как в D1).
4) Мебель и техника:
  - Мебель из шпона/МДФ высокого класса
  - Серийная мебель премиум-брендов (например, B&B Italia, Poliform).
  - Встроенная техника (Miele, Gaggenau), скрытые ручки/фасады.
  - VRF-кондиционирование
5) Освещение:
  - Дизайнерские светильники (Flos, Artemide), скрытая LED-подсветка.
  - Базовый умный дом (управление светом через приложение).
ИСКЛЮЧАЮЩИЕ ПРИЗНАКИ (если есть ХОТЯ БЫ ОДИН – это НЕ D0):
- Стены: Обои масс-маркет, гладкая покраска без дизайна, пластиковые/ПВХ-панели.
- Пол: Ламинат, линолеум, дешевая плитка с видимыми швами.
- Потолок: Побелка, натяжные потолки без интеграции света, открытые коммуникации
- Мебель: Мебель IKEA, Hoff; видимые провода, дешевые пластиковые фасады.
- Освещение: Люстры/бра масс-маркет, отсутствие автоматизации.

ГЛАВНЫЙ КРИТЕРИЙ [D0]: Полностью реализованный дизайн-проект с качественными материалами и продуманными инженерными решениями, но без ЭЛИТНЫХ/ЭКСКЛЮЗИВНЫХ элементов.
ПРИМЕР: студийный ремонт по проекту.

ЛЮБОЕ ДРУГОЕ ИЗОБРАЖЕНИЕ, не соответствующее описанию, классифицируй как [UNKNOWN].
Если есть сомнения – помечай как [UNKNOWN].
Отвечай ТОЛЬКО меткой класса [D0]/[UNKNOWN] и цифрой уверенности (1-10), без пояснений.
"""


D1_ONLY_FILTER_PROMPT_RU = """
Проанализируй фотографию помещения. Твоя задача — отфильтровать изображения класса [D1] (Эксклюзив / Luxury) от всего остального [UNKNOWN].
В ответе укажи ТОЛЬКО метку класса [D1]/[UNKNOWN] и уровень уверенности (например: "D1 9").

[D1] Эксклюзив (Luxury):
Отличительные черты:
- Элитные материалы: book-match камень, редкие породы дерева, натуральный шпон, металлические вставки (латунь, бронза).
- Высокотехнологичные решения: полная автоматизация (KNX, умный дом), скрытая подсветка, мультирум-аудиосистемы (7.1 и выше).
- Индивидуальный дизайн: авторская мебель (bespoke), сложные многоуровневые потолки, эксклюзивные декоративные элементы (например, ручная роспись).
- Премиальная отделка: мраморные/гранитные поверхности, панорамные окна, скрытые двери, бесшовные покрытия.
Исключающие признаки (если есть ХОТЯ БЫ ОДИН – это НЕ D1):
- Обычные материалы (ламинат, плитка без эксклюзивного дизайна, ЛДСП).
- Стандартная серийная мебель (IKEA, масс-маркет).
- Отсутствие сложных дизайнерских решений (простые ровные стены, базовые потолки).
- Нет признаков автоматизации или премиальной техники.

ГЛАВНЫЙ КРИТЕРИЙ [D1]: Комплексное сочетание элитных материалов, инновационных технологий и уникального дизайна.
ПРИМЕР: Резиденции с дизайнерской отделкой, luxury-апартаменты, бутик-отели.

ЛЮБОЕ ДРУГОЕ ИЗОБРАЖЕНИЕ, не соответствующее описанию, классифицируй как [UNKNOWN].
Если есть сомнения – помечай как [UNKNOWN]
Отвечай ТОЛЬКО меткой класса [D1]/[UNKNOWN] и цифрой уверенности (1-10), без пояснений.
"""

# ==========================================================================================================================================
# ==========================================================================================================================================


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
- Functional utilities (electrical/plumbing)
- Worn/damaged flooring (laminate/linoleum/parquet/floorboards)
- Peeling/swollen baseboards or ceiling finishes
- Faded, peeling, or cracked paint/tile
- Outdated/damaged furniture/doors/windows
- Faded or peeling wallpaper
- Cracks in paint/plaster/tile
- Functional but outdated plumbing
- Worn bathtub/shower/sink/toilet
- Example: Housing after 5-7 years of use

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


TRASH_FILTER_PROMPT_EN = """
You're a real estate photo analysis expert. Determine if the image is trash (unsuitable for interior assessment). Respond STRICTLY with "1" (trash) or "0" (interior).

Trash categories (always answer "1"):

1. A floor plan
2. Building facades (exterior views)
3. Stairwells/hallways
4. Yards/parking lots/streets
5. Floorplans/blueprints (top-down)
6. Documents/screenshots
7. Close-ups of people/animals
8. Blank white images (upload artifacts)
9. Blurry/overexposed shots
10. Furniture parts without room context
11. Close-up windows/doors without room view

If NONE apply - answer "0". No explanations.
"""

A0_ONLY_FILTER_PROMPT_EN = """
Analyze the photo of a rough construction state premises. Your task is to help filter dataset images into class [A0] versus everything else [UNKNOWN].
In response, specify ONLY the class label [A0]/[UNKNOWN] and confidence score (e.g.: "A0 8").

[A0] Unfinished (bare construction state):
Distinctive features:
- Concrete/brick/unprocessed walls without putty, plaster or decorative finishes
- Floor with rough screed or bare concrete, without any covering (tiles, laminate, etc.)
- Ceiling as bare concrete slab without finishing, possibly with exposed utilities, wiring or mounting hooks
- Exposed engineering systems (pipes, wires, cable channels)

Exclusion criteria (if ANY SINGLE ONE is present - it's NOT A0):
- Presence of any furniture, kitchen units, appliances, radiators, doors, plumbing fixtures
- Presence of interior doors, window sills
- Presence of even partial finishing (e.g., only walls are plastered or self-leveling compound is applied to floor)

CORE CRITERIA [A0]: Complete absence of finishing layers on all surfaces.
EXAMPLE: Unfinished construction shell without any finishing.

ANY OTHER IMAGE not matching this description should be classified as [UNKNOWN]. If any exclusion criteria are present, classify as [UNKNOWN].
Respond ONLY with the class label [A0]/[UNKNOWN] and confidence score, without explanations.
"""


D0_ONLY_FILTER_PROMPT_EN = """
Analyze the photo of a room. Your task is to filter images of class [D0] (Premium renovation) from everything else [UNKNOWN].
In your response, specify ONLY the class label [D0]/[UNKNOWN] and confidence level (e.g., "D0 9").

[D0] Designer renovation (European-quality renovation):
DISTINCTIVE FEATURES:
1) Walls:
  - Decorative plaster (Venetian, microcement), textured paint.
  - Natural materials: stone panels (no book-match), wood panels (veneer, solid wood).
  - Moderate decor allowed: moldings, niches with lighting.
2) Floor:
  - Natural materials: engineered hardwood (oak, ash), stone-look porcelain tile.
  - Seamless coatings: decorative epoxy floors.
3) Ceiling:
  - Suspended constructions (drywall, slatted systems) with spot/linear lighting.
  - Minimalist multi-level ceilings (without complex designs like in D1).
4) Furniture and appliances:
  - High-grade veneer/MDF furniture
  - Premium serial furniture (e.g., B&B Italia, Poliform).
  - Built-in appliances (Miele, Gaggenau), hidden handles/fronts.
  - VRF air conditioning
5) Lighting:
  - Designer fixtures (Flos, Artemide), hidden LED lighting.
  - Basic smart home (light control via app).

EXCLUSION CRITERIA (if ANY SINGLE ONE is present → it's NOT D0):
- Walls: Mass-market wallpaper, plain undecorated paint, plastic/PVC panels.
- Floor: Laminate, linoleum, cheap tile with visible grout lines.
- Ceiling: Whitewash, stretch ceilings without integrated lighting, exposed utilities.
- Furniture: IKEA, Hoff furniture; visible wires, cheap plastic fronts.
- Lighting: Mass-market chandeliers/sconces, lack of automation.

MAIN CRITERION [D0]: Fully implemented design project with quality materials and engineered solutions, but WITHOUT ELITE/EXCLUSIVE elements.
EXAMPLE: Studio apartment renovated by design project.

ANY OTHER IMAGE not matching this description should be classified as [UNKNOWN].
When in doubt – mark as [UNKNOWN].
Respond ONLY with class label [D0]/[UNKNOWN] and confidence number (1-10), no explanations.
"""


D1_ONLY_FILTER_PROMPT_EN = """
Analyze the photo of the room. Your task is to filter the images of the class [D1] (Exclusive / Luxury) from everything else [UNKNOWN].
In the response, specify ONLY the class label [D1]/[UNKNOWN] and the confidence level (for example: "D1 9"), without explanation.

[D1] Exclusive (Luxury):
Distinctive features:
- Elite materials: book-match stone, rare wood species, natural veneer, metal inserts (brass, bronze).
- High-tech solutions: full automation (KNX, smart home), hidden lighting, multiroom audio systems (7.1 and higher).
- Individual design: bespoke furniture, complex multi-level ceilings, exclusive decorative elements (for example, hand-painted).
- Premium finishes: marble/granite surfaces, panoramic windows, hidden doors, seamless coatings.
Exclusion criteria (if ANY SINGLE ONE is present - it's NOT D1):
- Common materials (laminate, tiles without exclusive design, chipboard).
- Standard standard furniture (IKEA, mass market).
- The absence of complex design solutions (simple smooth walls, basic ceilings).
- There are no signs of automation or premium technology.

THE MAIN CRITERION [D1]: A comprehensive combination of high-end materials, innovative technologies and unique design.
EXAMPLE: Residences with designer finishes, luxury apartments, boutique hotels.

Classify ANY OTHER IMAGE that does not match the description as [UNKNOWN].
If in doubt, mark it as [UNKNOWN]
Answer ONLY with the class label [D1]/[UNKNOWN] and the confidence number (1-10), without explanation.
"""