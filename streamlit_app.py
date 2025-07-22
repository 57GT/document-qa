import streamlit as st
import itertools
import random
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import json
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass, asdict
from functools import lru_cache
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

st.set_page_config(page_title="Optimizador de Eventos", layout="wide")

# -------------------------------
# Estructuras de Datos
# -------------------------------


@dataclass
class Section:
    name: str
    seats: int
    price: float = 0.0


@dataclass
class Scenario:
    name: str
    sell_rate: float
    prices: List[float]
    sections: List[Section]
    total_revenue: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict:
        """Convierte el escenario a un diccionario serializable."""
        return {
            'name': self.name,
            'sell_rate': self.sell_rate,
            'prices': self.prices,
            'sections': [{'name': s.name, 'seats': s.seats} for s in self.sections],
            'total_revenue': self.total_revenue,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Scenario':
        """Crea un escenario desde un diccionario."""
        sections = [Section(name=s['name'], seats=s['seats'])
                    for s in data['sections']]
        return cls(
            name=data['name'],
            sell_rate=data['sell_rate'],
            prices=data['prices'],
            sections=sections,
            total_revenue=data['total_revenue'],
            timestamp=data['timestamp']
        )

# -------------------------------
# Configuración y Constantes
# -------------------------------


SELL_RATES = {
    "alta": 0.98,
    "moderada": 0.90,
    "baja": 0.85,
    "teatro": 0.60
}

PRESENTATION_TEMPLATES = {
    "executive": {"template": "plotly_white", "height": 600},
    "detailed": {"template": "plotly", "height": 800},
    "minimal": {"template": "plotly_white", "height": 400}
}

# -------------------------------
# Funciones de Caché y Optimización
# -------------------------------


@lru_cache(maxsize=128)
def calculate_revenue(prices: Tuple[float, ...], seats: Tuple[int, ...], sell_rate: float) -> float:
    """Calcula el ingreso total con caché para mejorar rendimiento."""
    return round(sum(p * s * sell_rate for p, s in zip(prices, seats)), 2)


@lru_cache(maxsize=64)
def generate_price_candidates(min_price: float, max_price: float, num_candidates: int = 9) -> List[float]:
    """Genera precios candidatos ENTEROS con caché para mejorar rendimiento."""
    min_price = int(round(min_price))
    max_price = int(round(max_price))
    if max_price <= min_price:
        return [float(min_price)]
    step = max((max_price - min_price) // (num_candidates - 1), 1)
    return [float(min_price + i * step) for i in range(num_candidates) if (min_price + i * step) <= max_price]

# -------------------------------
# Funciones de Generación de Precios
# -------------------------------


def generate_valid_combinations(sections: List[Section], scenario: str,
                                global_min: float, global_max: float,
                                margin_factor: float) -> List[List[float]]:
    candidates = []
    for i, _ in enumerate(sections):
        if i == 0:
            sec_candidates = [float(int(round(global_max)))]
        elif i == len(sections) - 1:
            sec_candidates = [float(int(round(global_min)))]
        else:
            sec_candidates = generate_price_candidates(global_min, global_max)
        candidates.append(sec_candidates)

    valid = []
    for combo in itertools.islice(itertools.product(*candidates), 100_000):
        # Convertir a enteros explícitamente
        combo_int = [int(round(p)) for p in combo]
        if all(combo_int[i] >= margin_factor * combo_int[i+1] for i in range(len(combo_int) - 1)):
            valid.append(combo_int)
    return valid


def heuristic_price_search(target: float, sections: List[Section],
                           global_min: float, global_max: float,
                           margin_factor: float, scenario: str) -> List[float]:
    """Búsqueda heurística optimizada de precios ENTEROS."""
    best_combo, best_diff = None, float('inf')
    sell_rate = SELL_RATES[scenario]

    for _ in range(5_000):  # Reducido para mejor rendimiento
        combo = [int(round(global_max))]
        prev_price = int(round(global_max))

        for _ in range(len(sections) - 2):
            min_price = max(int(round(global_min)), int(
                prev_price / margin_factor * 0.95))
            max_price = int(prev_price / margin_factor)
            if max_price < min_price:
                price = min_price
            else:
                price = random.randint(min_price, max_price)
            combo.append(price)
            prev_price = price

        combo.append(int(round(global_min)))

        if all(combo[i] >= margin_factor * combo[i+1] for i in range(len(combo) - 1)):
            revenue = calculate_revenue(tuple(combo),
                                        tuple(s.seats for s in sections),
                                        sell_rate)
            if abs(revenue - target) < best_diff:
                best_combo, best_diff = combo, abs(revenue - target)

    return best_combo

# -------------------------------
# Funciones de Recomendación de Precios
# -------------------------------


def generate_training_data(num_samples: int = 1000) -> List[Dict]:
    """Genera datos de entrenamiento sintéticos para el modelo de recomendación."""
    training_data = []

    # Generar diferentes configuraciones de secciones
    section_configs = [
        [{"name": "VIP", "seats": 100}, {"name": "Premium",
                                         "seats": 300}, {"name": "General", "seats": 600}],
        [{"name": "VIP", "seats": 150}, {"name": "Premium",
                                         "seats": 450}, {"name": "General", "seats": 900}],
        [{"name": "VIP", "seats": 200}, {"name": "Premium",
                                         "seats": 600}, {"name": "General", "seats": 1200}],
        [{"name": "VIP", "seats": 80}, {"name": "Premium",
                                        "seats": 240}, {"name": "General", "seats": 480}],
        [{"name": "VIP", "seats": 120}, {"name": "Premium",
                                         "seats": 360}, {"name": "General", "seats": 720}]
    ]

    for _ in range(num_samples):
        # Seleccionar una configuración aleatoria
        section_config = random.choice(section_configs)
        sections = [Section(name=s["name"], seats=s["seats"])
                    for s in section_config]

        # Generar parámetros aleatorios
        target = random.uniform(10000, 100000)
        sell_rate = random.choice(list(SELL_RATES.values()))
        global_min = random.uniform(50, 200)
        global_max = random.uniform(global_min * 3, global_min * 10)
        margin_factor = random.uniform(1.02, 1.30)

        # Generar combinaciones válidas
        combos = generate_valid_combinations(
            sections, "alta", global_min, global_max, margin_factor)

        if combos:
            # Seleccionar la mejor combinación
            best_combo = min(combos, key=lambda x: abs(calculate_revenue(
                tuple(x),
                tuple(s.seats for s in sections),
                sell_rate
            ) - target))

            # Calcular el ingreso real
            revenue = calculate_revenue(tuple(best_combo), tuple(
                s.seats for s in sections), sell_rate)

            # Agregar datos de entrenamiento para cada sección
            for i, (price, section) in enumerate(zip(best_combo, sections)):
                training_data.append({
                    'seats': section.seats,
                    'sell_rate': sell_rate,
                    'target_income': target,
                    'section_index': i,
                    'optimal_price': price,
                    'revenue': revenue
                })

    return training_data


def train_price_recommender(historical_data: List[Dict] = None, save_path: str = "price_recommender.joblib") -> RandomForestRegressor:
    """Entrena un modelo simple para recomendar precios."""
    # Si no se proporcionan datos históricos, generar datos de entrenamiento sintéticos
    if historical_data is None:
        historical_data = generate_training_data()

    X = []
    y = []

    for data in historical_data:
        features = [
            data['seats'],
            data['sell_rate'],
            data['target_income'],
            # Posición de la sección (0 para VIP, 1 para Premium, etc.)
            data['section_index']
        ]
        X.append(features)
        y.append(data['optimal_price'])

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, save_path)
    return model


def load_price_recommender(load_path: str = "price_recommender.joblib") -> RandomForestRegressor:
    """Carga el modelo de recomendación si existe, de lo contrario lo entrena."""
    if os.path.exists(load_path):
        return joblib.load(load_path)
    else:
        return train_price_recommender(save_path=load_path)


def get_price_recommendations(model, sections: List[Section], target: float, sell_rate: float) -> List[float]:
    """Obtiene recomendaciones de precios del modelo entrenado, REDONDEADAS a enteros."""
    recommendations = []

    for i, section in enumerate(sections):
        features = [
            section.seats,
            sell_rate,
            target,
            i
        ]
        recommended_price = model.predict([features])[0]
        recommendations.append(int(round(recommended_price)))

    return recommendations

# -------------------------------
# Funciones de Visualización
# -------------------------------


def create_interactive_price_chart(scenarios: Dict[str, Scenario], presentation_mode: str) -> go.Figure:
    """Crea un gráfico interactivo de precios usando Plotly y el modo de presentación."""
    template = PRESENTATION_TEMPLATES[presentation_mode]["template"]
    height = PRESENTATION_TEMPLATES[presentation_mode]["height"]
    fig = go.Figure()

    for scenario_name, scenario in scenarios.items():
        fig.add_trace(go.Scatter(
            x=[s.name for s in scenario.sections],
            y=scenario.prices,
            name=scenario_name,
            mode='lines+markers',
            hovertemplate='Sección: %{x}<br>Precio: $%{y:.0f} MXN<extra></extra>'
        ))

    fig.update_layout(
        title="Comparación de Precios por Escenario",
        xaxis_title="Sección",
        yaxis_title="Precio (MXN)",
        template=template,
        hovermode='x unified',
        height=height
    )

    return fig


def create_revenue_comparison_chart(scenarios: Dict[str, Scenario], presentation_mode: str) -> go.Figure:
    """Crea un gráfico de comparación de ingresos usando Plotly."""
    template = PRESENTATION_TEMPLATES[presentation_mode]["template"]
    height = PRESENTATION_TEMPLATES[presentation_mode]["height"]
    fig = go.Figure()

    for scenario_name, scenario in scenarios.items():
        ingresos = [p * s.seats * scenario.sell_rate
                    for p, s in zip(scenario.prices, scenario.sections)]

        fig.add_trace(go.Bar(
            x=[s.name for s in scenario.sections],
            y=ingresos,
            name=scenario_name,
            hovertemplate='Sección: %{x}<br>Ingreso: $%{y:,.0f} MXN<extra></extra>'
        ))

    fig.update_layout(
        title="Comparación de Ingresos por Sección",
        xaxis_title="Sección",
        yaxis_title="Ingreso (MXN)",
        template=template,
        barmode='group',
        height=height
    )

    return fig

# -------------------------------
# Funciones de Gestión de Escenarios
# -------------------------------


def export_scenario_to_json(scenario: Dict) -> str:
    """Exporta un escenario a formato JSON."""
    # Convertir el escenario a un diccionario serializable
    serializable_data = {}
    for name, scenario_obj in scenario.items():
        serializable_data[name] = {
            'name': scenario_obj.name,
            'sell_rate': scenario_obj.sell_rate,
            'prices': scenario_obj.prices,
            'sections': [{'name': s.name, 'seats': s.seats} for s in scenario_obj.sections],
            'total_revenue': scenario_obj.total_revenue,
            'timestamp': scenario_obj.timestamp
        }
    return json.dumps(serializable_data, indent=2)


def import_scenario_from_json(json_str: str) -> Dict:
    """Importa un escenario desde formato JSON."""
    try:
        data = json.loads(json_str)
        # Reconstruir los objetos Scenario
        scenarios = {}
        for name, scenario_data in data.items():
            sections = [Section(name=s['name'], seats=s['seats'])
                        for s in scenario_data['sections']]
            scenarios[name] = Scenario(
                name=scenario_data['name'],
                sell_rate=scenario_data['sell_rate'],
                prices=scenario_data['prices'],
                sections=sections,
                total_revenue=scenario_data['total_revenue'],
                timestamp=scenario_data['timestamp']
            )
        return scenarios
    except json.JSONDecodeError:
        st.error("Error al importar escenario: formato JSON inválido")
        return None
    except (KeyError, TypeError) as e:
        st.error(
            f"Error al importar escenario: datos incompletos o incorrectos - {str(e)}")
        return None

# -------------------------------
# Funciones de Análisis
# -------------------------------


def calculate_sensitivity_analysis(scenario: Scenario, variation_range: float = 0.1):
    """Calcula el análisis de sensibilidad para un escenario dado."""
    base_revenue = scenario.total_revenue
    variations = []

    for i, (price, section) in enumerate(zip(scenario.prices, scenario.sections)):
        # Variación positiva
        new_price = price * (1 + variation_range)
        new_revenue = calculate_revenue(
            tuple(scenario.prices[:i] + [new_price] + scenario.prices[i+1:]),
            tuple(s.seats for s in scenario.sections),
            scenario.sell_rate
        )
        variations.append({
            'Sección': section.name,
            'Variación': f'+{variation_range*100}%',
            'Impacto en Ingreso': f'{((new_revenue - base_revenue) / base_revenue * 100):.2f}%'
        })

        # Variación negativa
        new_price = price * (1 - variation_range)
        new_revenue = calculate_revenue(
            tuple(scenario.prices[:i] + [new_price] + scenario.prices[i+1:]),
            tuple(s.seats for s in scenario.sections),
            scenario.sell_rate
        )
        variations.append({
            'Sección': section.name,
            'Variación': f'-{variation_range*100}%',
            'Impacto en Ingreso': f'{((new_revenue - base_revenue) / base_revenue * 100):.2f}%'
        })

    return pd.DataFrame(variations)


def calculate_maximum_possible_revenue(sections: List[Section], global_max: float, sell_rate: float) -> float:
    """Calcula el ingreso máximo posible con el precio máximo y la mejor tasa de venta."""
    return sum(s.seats * global_max * sell_rate for s in sections)


def calculate_minimum_possible_revenue(sections: List[Section], global_min: float, sell_rate: float) -> float:
    """Calcula el ingreso mínimo posible con el precio mínimo y la peor tasa de venta."""
    return sum(s.seats * global_min * sell_rate for s in sections)

# -------------------------------
# Función Principal
# -------------------------------


def main():

    # Inicialización de variables de sesión
    if 'current_scenario' not in st.session_state:
        st.session_state.current_scenario = None
    if 'presentation_mode' not in st.session_state:
        st.session_state.presentation_mode = "executive"
    if 'sensitivity_variation' not in st.session_state:
        st.session_state.sensitivity_variation = 5  # Valor predeterminado
    if 'price_recommender' not in st.session_state:
        # Entrenar el modelo de recomendación al iniciar la aplicación
        st.session_state.price_recommender = load_price_recommender()
    # Inicializar valores de configuración
    if 'target' not in st.session_state:
        st.session_state.target = 50000.0
    if 'num_sections' not in st.session_state:
        st.session_state.num_sections = 3
    if 'margin' not in st.session_state:
        st.session_state.margin = 5
    if 'global_min' not in st.session_state:
        st.session_state.global_min = 50.0
    if 'global_max' not in st.session_state:
        st.session_state.global_max = 5000.0
    if 'section_names' not in st.session_state:
        st.session_state.section_names = [f"Sección {i+1}" for i in range(3)]
    if 'section_seats' not in st.session_state:
        st.session_state.section_seats = [500] * 3

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración Global")

        # Modo de presentación
        presentation_mode = st.selectbox(
            "Modo de Presentación",
            ["executive", "detailed", "minimal"],
            format_func=lambda x: {
                "executive": "Ejecutivo",
                "detailed": "Detallado",
                "minimal": "Minimalista"
            }[x]
        )
        st.session_state.presentation_mode = presentation_mode

        # Configuración básica
        target = st.number_input(
            "Ingreso Objetivo (MXN)",
            min_value=1000.0,
            value=st.session_state.target,
            step=1000.0,
            format="%.2f",
            key="target_input"
        )
        st.session_state.target = target

        num_sections = st.number_input(
            "Número de Secciones",
            min_value=2,
            value=st.session_state.num_sections,
            step=1,
            key="num_sections_input"
        )
        st.session_state.num_sections = num_sections

        margin = st.slider(
            "Margen Mínimo entre Secciones (%)",
            2, 30, st.session_state.margin,
            key="margin_input"
        )
        st.session_state.margin = margin

        global_min = st.number_input(
            "Precio Mínimo (MXN)",
            value=st.session_state.global_min,
            format="%.2f",
            key="global_min_input"
        )
        st.session_state.global_min = global_min

        global_max = st.number_input(
            "Precio Máximo (MXN)",
            value=st.session_state.global_max,
            min_value=global_min + 1,
            format="%.2f",
            key="global_max_input"
        )
        st.session_state.global_max = global_max

        # Gestión de escenarios
        st.header("💾 Gestión de Escenarios")

        # Exportar escenario actual
        if st.session_state.current_scenario:
            json_str = export_scenario_to_json(
                st.session_state.current_scenario)
            st.download_button(
                "📤 Exportar Escenario Actual",
                json_str,
                f"escenario_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                "application/json"
            )

        # Importar escenario
        uploaded_file = st.file_uploader("Importar Escenario", type=['json'])
        if uploaded_file:
            json_str = uploaded_file.getvalue().decode()
            imported_scenario = import_scenario_from_json(json_str)
            if imported_scenario:
                st.session_state.current_scenario = imported_scenario
                # Sincronizar secciones
                first_scenario = next(iter(imported_scenario.values()))
                n = len(first_scenario.sections)
                st.session_state.num_sections = n
                st.session_state.section_names = [
                    s.name for s in first_scenario.sections]
                st.session_state.section_seats = [
                    s.seats for s in first_scenario.sections]
                st.success(
                    "Escenario importado correctamente y secciones sincronizadas")
                st.rerun()

    # Contenido principal
    st.title("🎟️ Optimizador de Precios para Eventos")

    # Configuración de secciones
    sections = []
    st.header("📋 Configuración de Secciones")

    # Sincronizar arrays si cambia el número de secciones
    if len(st.session_state.section_names) != st.session_state.num_sections:
        if len(st.session_state.section_names) < st.session_state.num_sections:
            st.session_state.section_names += [f"Sección {i+1}" for i in range(
                len(st.session_state.section_names), st.session_state.num_sections)]
            st.session_state.section_seats += [500] * (
                st.session_state.num_sections - len(st.session_state.section_seats))
        else:
            st.session_state.section_names = st.session_state.section_names[
                :st.session_state.num_sections]
            st.session_state.section_seats = st.session_state.section_seats[
                :st.session_state.num_sections]

    # Usar columnas para disposición más compacta
    cols = st.columns(min(3, st.session_state.num_sections))
    for i in range(st.session_state.num_sections):
        col_idx = i % len(cols)
        with cols[col_idx]:
            name = st.text_input(
                f"Nombre",
                value=st.session_state.section_names[i],
                key=f"name_{i}"
            )
            st.session_state.section_names[i] = name

            seats = st.number_input(
                "Asientos",
                100, 10000,
                st.session_state.section_seats[i],
                key=f"seats_{i}"
            )
            st.session_state.section_seats[i] = seats

            sections.append(Section(name=name, seats=seats))

    # Mostrar análisis de ingresos posibles
    st.header("💰 Análisis de Ingresos Posibles")
    col1, col2, col3 = st.columns(3)
    with col1:
        max_revenue = calculate_maximum_possible_revenue(
            sections, st.session_state.global_max, max(SELL_RATES.values()))
        st.metric(
            "Ingreso Máximo Posible",
            f"${max_revenue:,.2f}",
            "Con precio máximo y mejor tasa de venta"
        )
    with col2:
        min_revenue = calculate_minimum_possible_revenue(
            sections, st.session_state.global_min, min(SELL_RATES.values()))
        st.metric(
            "Ingreso Mínimo Posible",
            f"${min_revenue:,.2f}",
            "Con precio mínimo y peor tasa de venta"
        )
    with col3:
        st.metric(
            "Rango de Ingresos",
            f"${max_revenue - min_revenue:,.2f}",
            "Diferencia entre máximo y mínimo"
        )

    # Optimización y visualización
    margin_factor = 1 + (st.session_state.margin / 100)
    scenarios = {
        "Alta Demanda": "alta",
        "Demanda Media": "moderada",
        "Baja Demanda": "baja",
        "Demanda Teatro": "teatro"
    }
    all_scenarios = {}
    # Progreso
    progress_bar = st.progress(0)
    # Crear tabs para cada escenario
    scenario_tabs = st.tabs([f"📊 {name}" for name in scenarios.keys()])
    for tab, (scenario_name, scenario_code) in zip(scenario_tabs, scenarios.items()):
        with tab:
            combos = generate_valid_combinations(
                sections, scenario_code,
                st.session_state.global_min, st.session_state.global_max, margin_factor
            )
            if not combos:
                st.warning("Usando algoritmo de aproximación...")
                best_combo = heuristic_price_search(
                    st.session_state.target, sections, st.session_state.global_min,
                    st.session_state.global_max, margin_factor, scenario_code
                )
                combos = [best_combo] if best_combo else []
            if combos:
                sell_rate = SELL_RATES[scenario_code]
                top_combos = sorted(
                    combos,
                    key=lambda x: abs(calculate_revenue(
                        tuple(x),
                        tuple(s.seats for s in sections),
                        sell_rate
                    ) - st.session_state.target)
                )[:3]
                scenario = Scenario(
                    name=scenario_name,
                    sell_rate=sell_rate,
                    prices=top_combos[0],
                    sections=sections,
                    total_revenue=calculate_revenue(
                        tuple(top_combos[0]),
                        tuple(s.seats for s in sections),
                        sell_rate
                    ),
                    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )
                all_scenarios[scenario_name] = scenario
                # Mostrar solo el gráfico principal en minimal y executive
                st.plotly_chart(
                    create_interactive_price_chart(
                        {scenario_name: scenario}, st.session_state.presentation_mode),
                    use_container_width=True
                )
                # Mostrar métricas y detalles solo si no es minimal
                if st.session_state.presentation_mode != 'minimal':
                    st.subheader("Opciones de Precios")
                    for idx, combo in enumerate(top_combos, 1):
                        revenue = calculate_revenue(
                            tuple(combo),
                            tuple(s.seats for s in sections),
                            sell_rate
                        )
                        if idx == 1:
                            st.markdown(
                                f"**Opción {idx} - Mejor aproximación al ingreso objetivo**")
                        elif idx == 2:
                            st.markdown(
                                f"**Opción {idx} - Segunda mejor aproximación**")
                        else:
                            st.markdown(
                                f"**Opción {idx} - Tercera mejor aproximación**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Ingreso Estimado",
                                f"${revenue:,.2f}",
                                delta=f"${revenue - st.session_state.target:,.2f}"
                            )
                        with col2:
                            st.write("Precios por sección:")
                            st.code(" | ".join(f"${int(p):,}" for p in combo))
                    st.subheader("🤖 Recomendaciones de IA")
                    st.info(
                        "Estas recomendaciones están basadas en un modelo de aprendizaje automático entrenado con datos sintéticos.")
                    recommended_prices = get_price_recommendations(
                        st.session_state.price_recommender,
                        sections,
                        st.session_state.target,
                        sell_rate
                    )
                    recommended_revenue = calculate_revenue(
                        tuple(recommended_prices),
                        tuple(s.seats for s in sections),
                        sell_rate
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Ingreso Estimado con Recomendaciones",
                            f"${recommended_revenue:,.2f}",
                            delta=f"${recommended_revenue - st.session_state.target:,.2f}"
                        )
                    with col2:
                        st.write("Precios recomendados por sección:")
                        st.code(" | ".join(
                            f"${int(p):,}" for p in recommended_prices))
                    best_revenue = calculate_revenue(
                        tuple(top_combos[0]),
                        tuple(s.seats for s in sections),
                        sell_rate
                    )
                    if abs(recommended_revenue - st.session_state.target) < abs(best_revenue - st.session_state.target):
                        st.success(
                            "¡Las recomendaciones de IA están más cerca del ingreso objetivo que la mejor opción encontrada!")
                    else:
                        st.info(
                            "La mejor opción encontrada está más cerca del ingreso objetivo que las recomendaciones de IA.")
                    # Análisis de sensibilidad solo en modo detailed
                    if st.session_state.presentation_mode == 'detailed':
                        st.subheader("Análisis de Sensibilidad")
                        sensi = calculate_sensitivity_analysis(scenario)
                        st.dataframe(sensi)
            else:
                st.error("No se encontraron combinaciones válidas")
        progress_bar.progress(
            (list(scenarios.keys()).index(scenario_name) + 1) / len(scenarios))
    if len(all_scenarios) > 1:
        st.header("📊 Comparación de Escenarios")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                create_interactive_price_chart(
                    all_scenarios, st.session_state.presentation_mode),
                use_container_width=True
            )
        with col2:
            st.plotly_chart(
                create_revenue_comparison_chart(
                    all_scenarios, st.session_state.presentation_mode),
                use_container_width=True
            )
    st.session_state.current_scenario = all_scenarios
    # Pedir nombre del show antes de descargar Excel
    if all_scenarios:
        show_name = st.text_input(
            "Nombre del Show para el reporte Excel:", value="Show")
        if show_name:
            excel_file = generate_excel_report(
                all_scenarios, st.session_state.presentation_mode, show_name)
            st.success("✅ Optimización completada")
            st.download_button(
                "📥 Descargar Reporte Excel",
                excel_file,
                f"Reporte_{show_name}_{st.session_state.presentation_mode}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )


def generate_excel_report(scenarios: Dict[str, Scenario], presentation_mode: str, show_name: str) -> BytesIO:
    """Genera un reporte Excel detallado."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        sheet_names = set()
        for scenario_name, scenario in scenarios.items():
            # Preparar datos según el modo
            data = {
                'Sección': [s.name for s in scenario.sections],
                'Precio Recomendado': [int(p) for p in scenario.prices],
                'Asientos Disponibles': [s.seats for s in scenario.sections],
            }
            if presentation_mode != 'minimal':
                data['Tasa de Venta'] = [
                    round(scenario.sell_rate, 2)] * len(scenario.sections)
                data['Asientos Vendidos'] = [
                    int(s.seats * scenario.sell_rate) for s in scenario.sections]
                data['Ingreso (MXN)'] = [int(p * s.seats * scenario.sell_rate)
                                         for p, s in zip(scenario.prices, scenario.sections)]
                total_revenue = sum(data['Ingreso (MXN)'])
                if total_revenue > 0:
                    data['% Contribución'] = [
                        round(r / total_revenue, 4) for r in data['Ingreso (MXN)']]
                else:
                    data['% Contribución'] = [0 for _ in data['Ingreso (MXN)']]
            df = pd.DataFrame(data)
            # Unicidad de nombres de hoja
            base_name = f"{scenario_name[:20]}_{presentation_mode}"
            sheet_name = base_name
            suffix = 1
            while sheet_name in sheet_names:
                sheet_name = f"{base_name}_{suffix}"
                suffix += 1
            sheet_names.add(sheet_name)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            money_format = workbook.add_format({'num_format': '$#,##0'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            worksheet.set_column('A:A', 20)
            worksheet.set_column('B:B', 20, money_format)
            worksheet.set_column('C:C', 20)
            if presentation_mode != 'minimal':
                worksheet.set_column('D:D', 15, percent_format)
                worksheet.set_column('E:E', 20)
                worksheet.set_column('F:F', 20, money_format)
                worksheet.set_column('G:G', 20, percent_format)
            # Análisis de sensibilidad solo en modo detailed
            if presentation_mode == 'detailed':
                sensi = calculate_sensitivity_analysis(scenario)
                # Sufijo corto y truncado para hoja de sensibilidad
                sensi_suffix = '_sens'
                max_base_len = 31 - len(sensi_suffix)
                sensi_sheet_base = sheet_name[:max_base_len]
                sensi_sheet_name = sensi_sheet_base + sensi_suffix
                sensi_suffix_count = 1
                while sensi_sheet_name in sheet_names:
                    # Si hay colisión, agregar número
                    extra = f"_{sensi_suffix_count}"
                    sensi_sheet_base = sheet_name[:max_base_len - len(extra)]
                    sensi_sheet_name = sensi_sheet_base + sensi_suffix + extra
                    sensi_suffix_count += 1
                sheet_names.add(sensi_sheet_name)
                sensi.to_excel(
                    writer, sheet_name=sensi_sheet_name, index=False)
    output.seek(0)
    return output


def check_password():
    password = st.text_input("Contraseña", type="password")
    if password == st.secrets["password"]:
        st.session_state["authenticated"] = True
        st.success("Acceso concedido.")
        return True
    else:
        st.session_state["authenticated"] = False
        if password != "":
            st.error("Acceso denegado.")
        return False


if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    if not check_password():
        st.stop()

if __name__ == "__main__":
    main()
