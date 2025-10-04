# -*- coding: utf-8 -*-
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

# ----------------------------
# Конфигурация
# ----------------------------
TELEGRAM_BOT_TOKEN = ""
CLOUD_API_KEY = ''

DATA_FILE = "df_fin.csv"
META_FILE = "top_clusters_meta.csv"
LAST_UPDATE_FILE = "last_update.txt"
RADAR_CACHE_FILE = "radar_cache.json"
TRUSTED_SOURCES_FILE = "trusted_sources.json"

client = OpenAI(
    api_key=CLOUD_API_KEY,
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
    if not os.path.exists(LAST_UPDATE_FILE):
        return False
    with open(LAST_UPDATE_FILE, "r") as f:
        last = f.read().strip()
    if last != today_iso():
        return False
    if not os.path.exists(DATA_FILE) or not os.path.exists(META_FILE):
        return False
    return True

def mark_updated_today():
    with open(LAST_UPDATE_FILE, "w") as f:
        f.write(today_iso())
    # Сбрасываем кэш при обновлении данных
    for f in [RADAR_CACHE_FILE, TRUSTED_SOURCES_FILE]:
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
        data = json.loads(fix_json(output))
        trusted = set(data.get("trusted_sources", []))
        with open(TRUSTED_SOURCES_FILE, "w", encoding="utf-8") as f:
            json.dump({"trusted_sources": list(trusted)}, f, indent=2, ensure_ascii=False)
        return trusted
    except Exception as e:
        logger.error(f"Ошибка генерации trusted_sources: {e}")
        return {"reuters.com", "bloomberg.com", "ft.com", "wsj.com", "cnbc.com"}

# ----------------------------
# Сохранение метаданных топ-кластеров с улучшенным hotness
# ----------------------------
def save_top_clusters_meta(df_fin, output_file=META_FILE):
    logger.info("🔍 Запуск кластеризации...")
    # Преобразуем даты
    df_fin['V21DATE_dt'] = pd.to_datetime(df_fin['V21DATE'], format='%Y%m%d%H%M%S', errors='coerce')
    df_fin = df_fin.dropna(subset=['V21DATE_dt']).copy()

    # Извлекаем слова из URL для семантики
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

    # Векторизация и кластеризация
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df_fin['semantic_text'])
    clustering = DBSCAN(eps=0.01, min_samples=2, metric='cosine').fit(tfidf_matrix)
    df_fin['semantic_cluster'] = clustering.labels_

    # Загружаем trusted sources
    if os.path.exists(TRUSTED_SOURCES_FILE):
        with open(TRUSTED_SOURCES_FILE, "r", encoding="utf-8") as f:
            trusted_sources = set(json.load(f)["trusted_sources"])
    else:
        trusted_sources = generate_trusted_sources()

    # Берём топ-10 кластеров по размеру
    top_clusters = df_fin['semantic_cluster'].value_counts().head(10)
    meta_rows = []

    for cluster_id in top_clusters.index:
        if cluster_id == -1:
            continue

        cluster_data = df_fin[df_fin['semantic_cluster'] == cluster_id].copy()
        cluster_data = cluster_data.sort_values('V21DATE_dt')

        urls = cluster_data['V2DOCUMENTIDENTIFIER'].dropna().tolist()
        dates = cluster_data['V21DATE_dt'].dropna().tolist()
        sources = cluster_data['V2SOURCECOMMONNAME'].dropna().tolist()

        if not dates or not urls:
            continue

        # Извлечение сущностей
        themes = cluster_data['V1THEMES'].dropna().str.split(';').explode().value_counts().head(10).index.tolist()
        persons = cluster_data['V1PERSONS'].dropna().str.split(';').explode().value_counts().head(5).index.tolist()
        orgs = cluster_data['V1ORGANIZATIONS'].dropna().str.split(';').explode().value_counts().head(5).index.tolist()

        start_date = min(dates)
        end_date = max(dates)

        # === НОВАЯ МЕТРИКА ГОРЯЧЕСТИ НА ОСНОВЕ ВСПЛЕСКА ===
        cluster_data['hour'] = cluster_data['V21DATE_dt'].dt.floor('h')
        mentions_per_hour = cluster_data.groupby('hour').size()
        max_mentions_in_hour = mentions_per_hour.max() if not mentions_per_hour.empty else 0
        hotness = min(1.0, max_mentions_in_hour / 30.0)  # 30 упоминаний/час = максимум

        # Дополнительные метаданные
        num_sources = len(set(sources))

        meta_rows.append({
            "cluster_id": cluster_id,
            "urls": ";".join(urls[:10]),
            "themes": ";".join(themes[:5]),
            "persons": ";".join(persons[:5]),
            "orgs": ";".join(orgs[:5]),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hotness": round(hotness, 3),
            "num_sources": num_sources,
            "sources_list": sources
        })

    pd.DataFrame(meta_rows).to_csv(output_file, index=False)
    logger.info(f"✅ Сохранено {len(meta_rows)} топ-кластеров в {output_file}")

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
                first_theme.startswith('TAX_FNCACT') or
                'FINANCE' in first_theme or
                first_theme.startswith('MARKET_') or
                first_theme.startswith('RATING_') or
                first_theme.startswith('COMMODITY_') or
                first_theme.startswith('CURRENCY_')
            )

        df['is_financial'] = df['V1THEMES'].apply(starts_with_financial_theme)
        
        df_fin = df[df['is_financial']].copy()
        if df_fin.empty:
            raise Exception("Нет финансовых записей")
        df_fin.to_csv(DATA_FILE, index=False)
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
    if was_updated_today():
        try:
            df_fin = pd.read_csv(DATA_FILE)
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
            df_fin = pd.read_csv(DATA_FILE)
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
    # Отправка из кэша
    if os.path.exists(RADAR_CACHE_FILE):
        with open(RADAR_CACHE_FILE, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        await update.message.reply_text("📬 Отправляю кэшированный отчёт...")
        await asyncio.sleep(5)
        for item in cached_data:
            output = (
                f"🔥 <b>{item['headline']}</b>\n"
                f"📊 Горячесть: {item['hotness']}\n"
                f"⏱ Почему сейчас: {item['why_now']}\n"
                f"🏷 Сущности: {', '.join(item['entities'])}\n"
                f"📅 Хронология: {item['timeline']}\n"
                f"🔗 Источники: {' | '.join([f'<a href=\"{s}\">{i+1}</a>' for i, s in enumerate(item['sources']) if s])}\n\n"
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

    # Генерация нового отчёта
    if not os.path.exists(META_FILE):
        await update.message.reply_text("⚠️ Нет кэшированных данных. Отправьте /start.")
        return

    try:
        meta_df = pd.read_csv(META_FILE)
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
                analysis = json.loads(fix_json(raw_output))
            except Exception as e:
                logger.warning(f"Не удалось распарсить JSON: {e}")
                continue

            cache_entries.append(analysis)

            output = (
                f"🔥 <b>{analysis['headline']}</b>\n"
                f"📊 Горячесть: {analysis['hotness']}\n"
                f"⏱ Почему сейчас: {analysis['why_now']}\n"
                f"🏷 Сущности: {', '.join(analysis['entities'])}\n"
                f"📅 Хронология: {analysis['timeline']}\n"
                f"🔗 Источники: {' | '.join([f'<a href=\"{s}\">{i+1}</a>' for i, s in enumerate(analysis['sources']) if s])}\n\n"
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

        with open(RADAR_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_entries, f, ensure_ascii=False, indent=2)

        await update.message.reply_text("✅ Все новости отправлены и сохранены в кэш!")
    except Exception as e:
        logger.exception("Ошибка в /radar")
        await update.message.reply_text(f"❌ Ошибка: {str(e)[:200]}")

# ----------------------------
# Запуск
# ----------------------------

def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("radar", radar_command))
    logger.info("🚀 Бот запущен. Отправьте /start для загрузки данных.")
    application.run_polling()

if __name__ == "__main__":
    main()
