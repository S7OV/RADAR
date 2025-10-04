from config.config import parseConfig, Config
import os
import logging
import pandas as pd
import requests
import zipfile
import io
import json
import asyncio
from datetime import datetime, timezone, timedelta
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from bs4 import BeautifulSoup

conf = parseConfig()

client = OpenAI(
    api_key=conf.CLOUD_API_KEY,
    base_url="https://foundation-models.api.cloud.ru/v1"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# ----------------------------
# Вспомогательные функции
# ----------------------------

def today_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def was_updated_today():
    if not os.path.exists(conf.LAST_UPDATE_FILE):
        return False
    with open(conf.LAST_UPDATE_FILE, "r") as f:
        last = f.read().strip()
    if last != today_iso():
        return False
    if not os.path.exists(conf.DATA_FILE) or not os.path.exists(conf.META_FILE):
        return False
    return True

def mark_updated_today():
    with open(conf.LAST_UPDATE_FILE, "w") as f:
        f.write(today_iso())
    # Сбрасываем кэш при обновлении данных
    for f in [conf.RADAR_CACHE_FILE, conf.TRUSTED_SOURCES_FILE]:
        if os.path.exists(f):
            os.remove(f)

def extract_slug_words_from_url(url):
    match = re.search(r'/([^/]+?)(?:\.html?|\.php|\.asp|\.aspx|#|\?|/$|$)', url)
    if match:
        slug = match.group(1).lower()
        slug_clean = re.sub(r'\b\d+\b', '', slug)
        slug_clean = re.sub(r'[^\w-]', ' ', slug_clean)
        words = [w for w in re.split(r'[-\s]+', slug_clean) if w]
        if words:
            return ' '.join(words)
    return ''

def safe_get_text(url):
    try:
        response = requests.get(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; Bot/1.0)'},
            timeout=5
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        article_body = (
            soup.find('article') or
            soup.find('div', class_='post-body') or
            soup.find('div', class_='content') or
            soup.find('body')
        )
        if article_body:
            return article_body.get_text(strip=True)
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
    return ""

def escape_html(text):
    return (
        text.replace("&", "&amp;")
            .replace("<", "<")
            .replace(">", ">")
    )

def fix_json(text):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end != 0:
        return text[start:end]
    return text

def generate_full_analysis(headline, entities, sources, timeline_str, full_text_preview, hotness, cluster_id):
    sources_list = sources[:3]
    prompt = f"""
Ты — аналитик финансовых новостей. На основе данных ниже создай структурированный отчёт в формате JSON.

Требуемый формат (строго соблюдай!):
{{
  "headline": "Краткий заголовок (до 8 слов)",
  "hotness": {hotness},
  "why_now": "1–2 предложения: почему это важно СЕЙЧАС? Упомяни неожиданность, подтверждения, масштаб эффекта.",
  "entities": ["список", "ключевых", "активов", "стран", "секторов"],
  "sources": {json.dumps(sources_list)},
  "timeline": "{timeline_str}",
  "draft": {{
    "title": "Заголовок для заметки",
    "lead": "Лид-абзац (1–2 предложения)",
    "bullets": ["Пункт 1", "Пункт 2", "Пункт 3"],
    "quote": "Цитата или сноска (если нет — напиши «Нет подтверждённой цитаты»)"
  }},
  "dedup_group": "{cluster_id}"
}}

Контекст:
- Заголовок события: {headline}
- Сущности: {entities}
- Источники: {sources_list}
- Хронология: {timeline_str}
- Фрагмент текста: {full_text_preview[:600]}
"""
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        max_tokens=4000,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ----------------------------
# Генерация trusted_sources.json
# ----------------------------

def generate_trusted_sources():
    if not os.path.exists("unique_sources.txt"):
        logger.warning("Файл unique_sources.txt не найден. Используем базовый список.")
        return {
            "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "cnbc.com",
            "bbc.com", "economist.com", "ap.org", "nytimes.com", "rbc.ru",
            "interfax.ru", "tass.ru", "kommersant.ru", "abc.net.au", "9news.com.au"
        }

    with open("unique_sources.txt", "r", encoding="utf-8") as f:
        sources = [line.strip() for line in f if line.strip()][:200]

    prompt = f"""Ты — эксперт по медиа и финансовой журналистике.  
Определи, какие из перечисленных новостных источников являются **надёжными** для финансового анализа.

Критерии:
- Известное международное или национальное СМИ с репутацией точности.
- Не блог, не агрегатор без редакции, не сатира.

Верни ТОЛЬКО JSON:
{{"trusted_sources": ["reuters.com", "bloomberg.com", ...]}}

Список:
{chr(10).join('- ' + s for s in sources)}
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            max_tokens=2000,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        output = response.choices[0].message.content
        data = json.loads(str(fix_json(output)))
        trusted = set(data.get("trusted_sources", []))
        with open(conf.TRUSTED_SOURCES_FILE, "w", encoding="utf-8") as f:
            json.dump({"trusted_sources": list(trusted)}, f, indent=2, ensure_ascii=False)
        return trusted
    except Exception as e:
        logger.error(f"Ошибка генерации trusted_sources: {e}")
        return {"reuters.com", "bloomberg.com", "ft.com", "wsj.com", "cnbc.com"}

# ----------------------------
# Сохранение метаданных топ-кластеров с композитной метрикой горячести
# ----------------------------
def save_top_clusters_meta(df_fin, output_file=conf.META_FILE):
    logger.info("🔍 Запуск кластеризации...")
    # Преобразуем даты
    df_fin['V21DATE_dt'] = pd.to_datetime(df_fin['V21DATE'], format='%Y%m%d%H%M%S', errors='coerce')
    df_fin = df_fin.dropna(subset=['V21DATE_dt']).copy()

    # Извлекаем слова из URL
    df_fin['url_slug_words'] = df_fin['V2DOCUMENTIDENTIFIER'].apply(extract_slug_words_from_url)
    df_fin['semantic_text'] = (
        df_fin['V1THEMES'].fillna('') + ' ' +
        df_fin['V2ENHANCEDTHEMES'].fillna('') + ' ' +
        df_fin['V1PERSONS'].fillna('') + ' ' +
        df_fin['V2ENHANCEDPERSONS'].fillna('') + ' ' +
        df_fin['V1ORGANIZATIONS'].fillna('') + ' ' +
        df_fin['V2ENHANCEDORGANIZATIONS'].fillna('') + ' ' +
        df_fin['V1LOCATIONS'].fillna('') + ' ' +
        df_fin['V2ENHANCEDLOCATIONS'].fillna('') + ' ' +
        df_fin['V2EXTRASXML'].fillna('') + ' ' +
        df_fin['url_slug_words']
    )

    # Кластеризация
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df_fin['semantic_text'])
    clustering = DBSCAN(eps=0.01, min_samples=2, metric='cosine').fit(tfidf_matrix)
    df_fin['semantic_cluster'] = clustering.labels_

    # Trusted sources
    if os.path.exists(conf.TRUSTED_SOURCES_FILE):
        with open(conf.TRUSTED_SOURCES_FILE, "r", encoding="utf-8") as f:
            trusted_sources = set(json.load(f)["trusted_sources"])
    else:
        trusted_sources = generate_trusted_sources()

    # Обрабатываем ВСЕ кластеры (кроме шума)
    all_clusters = df_fin['semantic_cluster'].unique()
    meta_rows = []

    for cluster_id in all_clusters:
        if cluster_id == -1:
            continue

        cluster_data = df_fin[df_fin['semantic_cluster'] == cluster_id].copy()
        cluster_data = cluster_data.sort_values('V21DATE_dt')

        urls = cluster_data['V2DOCUMENTIDENTIFIER'].dropna().tolist()
        dates = cluster_data['V21DATE_dt'].dropna().tolist()
        sources = cluster_data['V2SOURCECOMMONNAME'].dropna().tolist()

        if not dates or not urls:
            continue

        # Сущности
        themes_series = cluster_data['V1THEMES'].dropna().str.split(';').explode()
        themes = themes_series.value_counts().head(10).index.tolist()
        persons = cluster_data['V1PERSONS'].dropna().str.split(';').explode().value_counts().head(5).index.tolist()
        orgs = cluster_data['V1ORGANIZATIONS'].dropna().str.split(';').explode().value_counts().head(5).index.tolist()

        start_date = min(dates)
        end_date = max(dates)
        duration_hours = (end_date - start_date).total_seconds() / 3600
        num_sources = len(set(sources))

        # === 1. Внезапность (25%) ===
        surprise_score = len(cluster_data) / (duration_hours + 1)
        surprise_score = min(1.0, surprise_score / 50.0)

        # === 2. Материальность (25%) ===
        financial_prefixes = {'ECON_', 'MARKET_', 'TAX_'}
        financial_theme_count = sum(
            1 for t in themes_series if any(t.startswith(p) for p in financial_prefixes)
        )
        materiality_score = financial_theme_count / len(themes_series) if len(themes_series) > 0 else 0
        materiality_score = min(1.0, materiality_score)

        # === 3. Скорость распространения (20%) ===
        spread_speed = num_sources / (duration_hours + 0.1)
        spread_speed = min(1.0, spread_speed / 10.0)

        # === 4. Масштаб — разнообразие тем (15%) ===
        theme_prefixes = {t.split('_')[0] for t in themes if '_' in t and t.split('_')[0].isupper()}
        breadth_score = min(1.0, len(theme_prefixes) / 4.0)

        # === 5. Достоверность (15%) ===
        trusted_count = sum(1 for s in sources if s and s.lower() in trusted_sources)
        trust_ratio = trusted_count / max(num_sources, 1)
        confirmation_ratio = min(1.0, num_sources / 5.0)
        credibility_score = (trust_ratio + confirmation_ratio) / 2

        # === Итоговая горячесть ===
        hotness = (
            0.25 * surprise_score +
            0.25 * materiality_score +
            0.20 * spread_speed +
            0.15 * breadth_score +
            0.15 * credibility_score
        )

        meta_rows.append({
            "cluster_id": cluster_id,
            "urls": ";".join(urls[:10]),
            "themes": ";".join(themes[:5]),
            "persons": ";".join(persons[:5]),
            "orgs": ";".join(orgs[:5]),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hotness": round(hotness, 3),
            "citation_index": len(cluster_data),  # ← новое поле
            "hotness_breakdown": json.dumps({
                "surprise": round(surprise_score, 3),
                "materiality": round(materiality_score, 3),
                "spread_speed": round(spread_speed, 3),
                "breadth": round(breadth_score, 3),
                "credibility": round(credibility_score, 3)
            }),
            "num_sources": num_sources,
            "sources_list": sources
        })
        
    # Ранжируем по горячести и сохраняем топ-10
    if meta_rows:
        meta_df = pd.DataFrame(meta_rows)
        meta_df = meta_df.sort_values('hotness', ascending=False).head(10)
        meta_df.to_csv(output_file, index=False)
        logger.info(f"✅ Сохранено {len(meta_df)} топ-кластеров по горячести в {output_file}")
    else:
        pd.DataFrame().to_csv(output_file, index=False)
        logger.info("⚠️ Нет валидных кластеров для сохранения.")

# ----------------------------
# Загрузка GDELT и обработка
# ----------------------------

def fetch_gdelt_and_save():
    logger.info("🔄 Загрузка GDELT за последние 24 часа...")
    try:
        master_url = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
        response = requests.get(master_url, timeout=10)
        lines = response.text.strip().split('\n')
        now_utc = datetime.now(timezone.utc)
        cutoff_time = now_utc - timedelta(hours=24)
        gkg_urls_24h = []
        for line in reversed(lines):
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            url = parts[2]
            if not url.endswith('.gkg.csv.zip'):
                continue
            filename = url.split('/')[-1]
            ts = filename.split('.')[0]
            try:
                file_time = datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                if file_time >= cutoff_time:
                    gkg_urls_24h.append(url)
                else:
                    break
            except:
                continue
        all_dfs = []
        for url in gkg_urls_24h:
            try:
                resp = requests.get(url, timeout=10)
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    fname = z.namelist()[0]
                    with z.open(fname) as f:
                        df = pd.read_csv(f, sep='\t', header=None, dtype=str, encoding='latin1')
                        all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Ошибка при загрузке {url}: {e}")
                continue
        if not all_dfs:
            raise Exception("Нет данных за последние 24 часа")
        df = pd.concat(all_dfs, ignore_index=True)
        gkg_columns = [
            "GKGRECORDID", "V21DATE", "V2SOURCECOLLECTIONIDENTIFIER", "V2SOURCECOMMONNAME",
            "V2DOCUMENTIDENTIFIER", "V1COUNTS", "V21COUNTS", "V1THEMES", "V2ENHANCEDTHEMES",
            "V1LOCATIONS", "V2ENHANCEDLOCATIONS", "V1PERSONS", "V2ENHANCEDPERSONS",
            "V1ORGANIZATIONS", "V2ENHANCEDORGANIZATIONS", "V15TONE", "V21ENHANCEDDATES",
            "V2GCAM", "V21SHARINGIMAGE", "V21RELATEDIMAGES", "V21SOCIALIMAGEEMBEDS",
            "V21SOCIALVIDEOEMBEDS", "V21QUOTATIONS", "V21ALLNAMES", "V21AMOUNTS",
            "V21TRANSLATIONINFO", "V2EXTRASXML"
        ]
        if len(df.columns) == len(gkg_columns):
            df.columns = gkg_columns
        all_themes = df['V1THEMES'].dropna()
        unique_themes = set()
        for theme_str in all_themes:
            themes = theme_str.split(';')
            for theme in themes:
                unique_themes.add(theme.strip())

        def starts_with_financial_theme(themes):
            if pd.isna(themes) or not isinstance(themes, str):
                return False
            first_theme = themes.split(';')[0].strip()
            return (
                first_theme.startswith('ECON_') or
                first_theme.startswith('MARKET_') or
                first_theme.startswith('RATING_') or
                first_theme.startswith('COMMODITY_') or
                first_theme.startswith('CURRENCY_') or
                first_theme == 'FINANCE_COMPANY' or
                first_theme == 'FINANCE_BANKING'
            )

        df['is_financial'] = df['V1THEMES'].apply(starts_with_financial_theme)
        
        df_fin = df[df['is_financial']].copy()
        if df_fin.empty:
            raise Exception("Нет финансовых записей")
        df_fin.to_csv(conf.DATA_FILE, index=False)
        save_top_clusters_meta(df_fin)
        mark_updated_today()
        logger.info(f"✅ Обработка завершена: {len(df_fin)} записей, кластеры сохранены.")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке GDELT: {e}")
        return False

# ----------------------------
# Обработчики команд
# ----------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        raise ValueError
    if was_updated_today():
        try:
            df_fin = pd.read_csv(conf.DATA_FILE)
            df_fin['V21DATE_dt'] = pd.to_datetime(df_fin['V21DATE'], format='%Y%m%d%H%M%S', errors='coerce')
            min_date = df_fin['V21DATE_dt'].min()
            max_date = df_fin['V21DATE_dt'].max()
            count = len(df_fin)
            period = f"{min_date.strftime('%d.%m %H:%M')} – {max_date.strftime('%d.%m %H:%M')}"
            await update.message.reply_text(
                f"✅ База уже обновлена сегодня.\n"
                f"Содержит {count:,} финансовых новостей за период с {period}.\n"
                f"Используйте /radar."
            )
        except Exception as e:
            await update.message.reply_text("✅ База уже обновлена сегодня. Используйте /radar.")
        return
    await update.message.reply_text("🔄 Запускаю загрузку и анализ финансовых новостей... (~1–2 мин)")
    success = fetch_gdelt_and_save()
    if success:
        try:
            df_fin = pd.read_csv(conf.DATA_FILE)
            df_fin['V21DATE_dt'] = pd.to_datetime(df_fin['V21DATE'], format='%Y%m%d%H%M%S', errors='coerce')
            min_date = df_fin['V21DATE_dt'].min()
            max_date = df_fin['V21DATE_dt'].max()
            count = len(df_fin)
            period = f"{min_date.strftime('%d.%m %H:%M')} – {max_date.strftime('%d.%m %H:%M')}"
            await update.message.reply_text(
                f"✅ Анализ завершён!\n"
                f"Загружено {count:,} финансовых новостей за период с {period}.\n"
                f"Используйте /radar для получения отчёта."
            )
        except:
            await update.message.reply_text("✅ Анализ завершён! Используйте /radar для получения отчёта.")
    else:
        await update.message.reply_text("❌ Не удалось обновить данные. Попробуйте позже.")

async def radar_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        raise ValueError
    # --- Отправка из кэша ---
    if os.path.exists(conf.RADAR_CACHE_FILE):
        with open(conf.RADAR_CACHE_FILE, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        await update.message.reply_text("📬 Отправляю кэшированный отчёт...")
        await asyncio.sleep(5)
        for item in cached_data:
            citation_index = item.get("citation_index", len(item.get("sources", [])))
            breakdown = item.get("hotness_breakdown", {})
            hotness_details = (
                f"🧩 Горячесть: внезапность={breakdown.get('surprise', 0):.2f}, "
                f"материальность={breakdown.get('materiality', 0):.2f}, "
                f"скорость={breakdown.get('spread_speed', 0):.2f}, "
                f"масштаб={breakdown.get('breadth', 0):.2f}, "
                f"достоверность={breakdown.get('credibility', 0):.2f}"
            )
            output = (
                f"🔥 <b>{item['headline']}</b>\n"
                f"📊 Горячесть: {item['hotness']} | Цитирование: {citation_index}\n"
                f"{hotness_details}\n"
                f"⏱ Почему сейчас: {item['why_now']}\n"
                f"🏷 Сущности: {', '.join(item['entities'])}\n"
                f"📅 Хронология: {item['timeline']}\n"
                f"🔗 Источники: {' | '.join([f'<a href=\"{s}\">{i+1}</a>' for i, s in enumerate(item['sources']) if s])}\n"
                f"<b>Черновик:</b>\n"
                f"<b>{item['draft']['title']}</b>\n"
                f"{item['draft']['lead']}\n"
                f"• {item['draft']['bullets'][0]}\n"
                f"• {item['draft']['bullets'][1]}\n"
                f"• {item['draft']['bullets'][2]}\n"
                f"<i>{item['draft']['quote']}</i>\n"
                f"🆔 Кластер: {item['dedup_group']}"
            )
            await update.message.reply_text(output, parse_mode="HTML", disable_web_page_preview=True)
            await asyncio.sleep(1)
        await update.message.reply_text("✅ Отчёт отправлен из кэша!")
        return

    # --- Генерация нового отчёта ---
    if not os.path.exists(conf.META_FILE):
        await update.message.reply_text("⚠️ Нет кэшированных данных. Отправьте /start.")
        return

    try:
        meta_df = pd.read_csv(conf.META_FILE)
        if meta_df.empty:
            await update.message.reply_text("Нет значимых новостей за сутки.")
            return
        meta_df = meta_df.sort_values('hotness', ascending=False).head(10)
        await update.message.reply_text(f"📬 Генерирую топ-{len(meta_df)} финансовых новостей...")

        cache_entries = []
        for _, top_row in meta_df.iterrows():
            urls = top_row['urls'].split(';')
            themes = top_row['themes'].split(';') if pd.notna(top_row['themes']) else []
            persons = top_row['persons'].split(';') if pd.notna(top_row['persons']) else []
            orgs = top_row['orgs'].split(';') if pd.notna(top_row['orgs']) else []
            entities = list(set(themes + persons + orgs))
            headline = themes[0] if themes else "Финансовое событие"
            timeline_str = f"{top_row['start_date'][:16]} → {top_row['end_date'][:16]}"
            full_text_preview = safe_get_text(urls[0]) or ""

            # Получаем компоненты горячести (если они есть)
            hotness_breakdown = {}
            citation_index = top_row.get('citation_index', len(urls))

            # Если в conf.META_FILE есть breakdown — используем
            if 'hotness_breakdown' in top_row and pd.notna(top_row['hotness_breakdown']):
                try:
                    hotness_breakdown = json.loads(top_row['hotness_breakdown'])
                except:
                    pass

            raw_output = generate_full_analysis(
                headline=headline,
                entities=entities,
                sources=urls,
                timeline_str=timeline_str,
                full_text_preview=full_text_preview,
                hotness=top_row['hotness'],
                cluster_id=top_row['cluster_id']
            )

            try:
                analysis = json.loads(str(fix_json(raw_output)))
            except Exception as e:
                logger.warning(f"Не удалось распарсить JSON: {e}")
                continue

            # Добавляем служебные поля для кэша
            analysis["citation_index"] = citation_index
            analysis["hotness_breakdown"] = hotness_breakdown
            cache_entries.append(analysis)

            # Формируем вывод
            hotness_details = (
                f"🧩 Горячесть: внезапность={hotness_breakdown.get('surprise', 0):.2f}, "
                f"материальность={hotness_breakdown.get('materiality', 0):.2f}, "
                f"скорость={hotness_breakdown.get('spread_speed', 0):.2f}, "
                f"масштаб={hotness_breakdown.get('breadth', 0):.2f}, "
                f"достоверность={hotness_breakdown.get('credibility', 0):.2f}"
            )
            output = (
                f"🔥 <b>{analysis['headline']}</b>\n"
                f"📊 Горячесть: {analysis['hotness']} | Цитирование: {citation_index}\n"
                f"{hotness_details}\n"
                f"⏱ Почему сейчас: {analysis['why_now']}\n"
                f"🏷 Сущности: {', '.join(analysis['entities'])}\n"
                f"📅 Хронология: {analysis['timeline']}\n"
                f"🔗 Источники: {' | '.join([f'<a href=\"{s}\">{i+1}</a>' for i, s in enumerate(analysis['sources']) if s])}\n"
                f"<b>Черновик:</b>\n"
                f"<b>{analysis['draft']['title']}</b>\n"
                f"{analysis['draft']['lead']}\n"
                f"• {analysis['draft']['bullets'][0]}\n"
                f"• {analysis['draft']['bullets'][1]}\n"
                f"• {analysis['draft']['bullets'][2]}\n"
                f"<i>{analysis['draft']['quote']}</i>\n"
                f"🆔 Кластер: {analysis['dedup_group']}"
            )
            await update.message.reply_text(output, parse_mode="HTML", disable_web_page_preview=True)
            await asyncio.sleep(1)

        # Сохраняем в кэш с новыми полями
        with open(conf.RADAR_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_entries, f, ensure_ascii=False, indent=2)

        await update.message.reply_text("✅ Все новости отправлены и сохранены в кэш!")

    except Exception as e:
        logger.exception("Ошибка в /radar")
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:200]}")

# ----------------------------
# Запуск
# ----------------------------

def main():
    application = Application.builder().token(conf.TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("radar", radar_command))
    logger.info("🚀 Бот запущен. Отправьте /start для загрузки данных.")
    application.run_polling()

if __name__ == "__main__":
    main()