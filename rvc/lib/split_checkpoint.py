#!/usr/bin/env python3
"""
Утилита для разделения единого чекпоинта на отдельные G/D файлы.

Использование:
    python split_checkpoint.py path/to/checkpoint.pth output/folder/
    
Создаст:
    output/folder/G_pretrain.pth
    output/folder/D_pretrain.pth
"""

import argparse
import os
import sys

import torch


def split_checkpoint(checkpoint_path: str, output_dir: str):
    """
    Разделяет единый чекпоинт на отдельные G и D файлы.
    
    Args:
        checkpoint_path: Путь к единому чекпоинту (checkpoint.pth)
        output_dir: Папка для сохранения G_pretrain.pth и D_pretrain.pth
    """
    if not os.path.exists(checkpoint_path):
        print(f"Ошибка: файл '{checkpoint_path}' не найден!")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Загрузка чекпоинта: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        print("Загрузка в небезопасном режиме...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Проверка формата
    if "generator" not in checkpoint or "discriminator" not in checkpoint:
        print("Ошибка: файл не является единым чекпоинтом нового формата!")
        print("Ожидается структура с ключами 'generator' и 'discriminator'.")
        sys.exit(1)

    epoch = checkpoint.get("epoch", 0)
    learning_rate = checkpoint.get("learning_rate", 1e-4)

    # Извлечение генератора
    g_checkpoint = {
        "model": checkpoint["generator"]["model"],
        "optimizer": checkpoint["generator"]["optimizer"],
        "iteration": epoch,
        "learning_rate": learning_rate,
    }

    # Извлечение дискриминатора
    d_checkpoint = {
        "model": checkpoint["discriminator"]["model"],
        "optimizer": checkpoint["discriminator"]["optimizer"],
        "iteration": epoch,
        "learning_rate": learning_rate,
    }

    # Сохранение
    g_path = os.path.join(output_dir, "G_pretrain.pth")
    d_path = os.path.join(output_dir, "D_pretrain.pth")

    torch.save(g_checkpoint, g_path)
    print(f"Сохранён: {g_path}")

    torch.save(d_checkpoint, d_path)
    print(f"Сохранён: {d_path}")

    print(f"\nГотово! Эпоха чекпоинта: {epoch}")
    print(f"Файлы сохранены в: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Разделение единого чекпоинта на отдельные G/D файлы для претрейнов.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Примеры использования:
            - python split_checkpoint.py logs/MyModel/checkpoint.pth ./pretrain_output/
            - python split_checkpoint.py checkpoint.pth .
        """,
    )
    parser.add_argument("checkpoint_path", type=str, help="Путь к единому чекпоинту (checkpoint.pth)")
    parser.add_argument("output_dir", type=str, help="Папка для сохранения G_pretrain.pth и D_pretrain.pth")

    args = parser.parse_args()
    split_checkpoint(args.checkpoint_path, args.output_dir)


if __name__ == "__main__":
    main()
